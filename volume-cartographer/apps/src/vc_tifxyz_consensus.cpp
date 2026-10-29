#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/core.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Per-point agreement between repeat traces of the same surface.
//
// GrowPatch adds a small random offset to every new point, so growing the same
// seed with different --rng-seed values gives slightly different patches, and
// in compressed regions whole areas end up on a neighbouring sheet in one run
// and not in another. This tool takes one of those runs as the reference and
// counts, for each of its grid points, how many of the other runs pass within
// the tolerance. Points every run agrees on are much more likely to be right
// than points only the reference reached.

static void usage(const char* prog)
{
    std::cout
        << "usage: " << prog << " <reference tifxyz> <other tifxyz>... [options]\n"
        << "  For every valid point of the reference segment, counts how many of the\n"
        << "  other segments pass within --tolerance voxels of it, and stores the count\n"
        << "  as a uint8 channel <name>.tif in the reference segment (0 for invalid\n"
        << "  points). The other segments are normally repeat runs of\n"
        << "  vc_grow_seg_from_seed from the same seed with different --rng-seed values.\n"
        << "\n"
        << "options:\n"
        << "  --tolerance F    distance in voxels that counts as agreement (default 3)\n"
        << "  --name NAME      channel name (default consensus)\n"
        << "  --min-agree N    with --output: keep only points at least N others agree on\n"
        << "  --output DIR     write that filtered copy as a new segment to DIR instead\n"
        << "                   of adding the channel to the reference\n";
}

int main(int argc, char* argv[])
{
    std::vector<fs::path> segments;
    float tolerance = 3.0f;
    std::string name = "consensus";
    int minAgree = -1;
    fs::path output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: " << arg << " needs a value\n";
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return EXIT_SUCCESS;
        } else if (arg == "--tolerance") {
            tolerance = std::stof(value());
        } else if (arg == "--name") {
            name = value();
        } else if (arg == "--min-agree") {
            minAgree = std::stoi(value());
        } else if (arg == "--output") {
            output = value();
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "error: unknown option " << arg << "\n";
            return EXIT_FAILURE;
        } else {
            segments.emplace_back(arg);
        }
    }

    if (segments.size() < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!(tolerance > 0.0f)) {
        std::cerr << "error: --tolerance must be positive\n";
        return EXIT_FAILURE;
    }
    if (output.empty() != (minAgree < 0)) {
        std::cerr << "error: --min-agree and --output go together\n";
        return EXIT_FAILURE;
    }
    const int others = static_cast<int>(segments.size()) - 1;
    if (minAgree > others) {
        std::cerr << "error: --min-agree " << minAgree << " but only " << others
                  << " other segments\n";
        return EXIT_FAILURE;
    }
    for (size_t i = 1; i < segments.size(); ++i) {
        std::error_code ec;
        if (fs::equivalent(segments[0], segments[i], ec)) {
            std::cerr << "error: " << segments[i] << " is the reference itself\n";
            return EXIT_FAILURE;
        }
    }

    std::unique_ptr<QuadSurface> reference;
    std::vector<SurfacePatchIndex::SurfacePtr> runs;
    try {
        reference = load_quad_from_tifxyz(segments[0]);
        for (size_t i = 1; i < segments.size(); ++i) {
            runs.push_back(std::make_shared<QuadSurface>(segments[i]));
        }
    } catch (const std::exception& e) {
        std::cerr << "error loading segments: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    SurfacePatchIndex index;
    index.setSamplingStride(1);
    index.rebuild(runs);
    if (index.surfaceCount() != runs.size()) {
        std::cerr << "error: indexed " << index.surfaceCount() << " of " << runs.size()
                  << " segments\n";
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f>* points = reference->rawPointsPtr();
    cv::Mat_<uint8_t> agree(points->size(), uint8_t(0));

    #pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < points->rows; ++r) {
        for (int c = 0; c < points->cols; ++c) {
            const cv::Vec3f& p = (*points)(r, c);
            if (p[0] == -1.f) {
                continue;
            }
            SurfacePatchIndex::PointQuery query;
            query.worldPoint = p;
            query.tolerance = tolerance;
            agree(r, c) = static_cast<uint8_t>(index.locateSurfaces(query).size());
        }
    }

    std::vector<size_t> histogram(others + 1, 0);
    size_t valid = 0;
    for (int r = 0; r < points->rows; ++r) {
        for (int c = 0; c < points->cols; ++c) {
            if ((*points)(r, c)[0] != -1.f) {
                ++histogram[agree(r, c)];
                ++valid;
            }
        }
    }
    std::cout << "valid points " << valid << ", agreeing runs within " << tolerance
              << " vx:\n";
    for (int k = 0; k <= others; ++k) {
        std::cout << "  " << k << " of " << others << ": " << std::setw(10) << histogram[k]
                  << "  (" << std::fixed << std::setprecision(1)
                  << 100.0 * histogram[k] / std::max<size_t>(valid, 1) << " %)\n";
    }

    reference->setChannel(name, agree);
    if (output.empty()) {
        reference->saveChannel(name);
        std::cout << "wrote " << (segments[0] / (name + ".tif")).string() << "\n";
        return EXIT_SUCCESS;
    }

    size_t kept = 0;
    for (int r = 0; r < points->rows; ++r) {
        for (int c = 0; c < points->cols; ++c) {
            cv::Vec3f& p = (*points)(r, c);
            if (p[0] == -1.f) {
                continue;
            }
            if (agree(r, c) < minAgree) {
                p = cv::Vec3f(-1.f, -1.f, -1.f);
            } else {
                ++kept;
            }
        }
    }
    if (kept == 0) {
        std::cerr << "error: no point has " << minAgree << " agreeing runs, nothing to write\n";
        return EXIT_FAILURE;
    }
    reference->invalidateCache();
    reference->saveAsNew(output);
    std::cout << "kept " << kept << " of " << valid << " points, wrote " << output.string()
              << "\n";
    return EXIT_SUCCESS;
}
