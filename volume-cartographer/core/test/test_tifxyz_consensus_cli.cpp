#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "vc_test.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
TEST_CASE("cli tests are POSIX-only") {}
#else

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

// Flat 8-row patch at height z with 4 voxel grid spacing, cols columns wide.
void write_tifxyz(const fs::path& dir, const std::string& id, float z, int cols = 8)
{
    constexpr int rows = 8;
    cv::Mat x(rows, cols, CV_32F);
    cv::Mat y(rows, cols, CV_32F);
    cv::Mat zs(rows, cols, CV_32F, cv::Scalar(z));
    for (int v = 0; v < rows; ++v) {
        for (int u = 0; u < cols; ++u) {
            x.at<float>(v, u) = 4.f * u;
            y.at<float>(v, u) = 4.f * v;
        }
    }
    fs::create_directories(dir);
    REQUIRE(cv::imwrite((dir / "x.tif").string(), x));
    REQUIRE(cv::imwrite((dir / "y.tif").string(), y));
    REQUIRE(cv::imwrite((dir / "z.tif").string(), zs));
    std::ofstream meta(dir / "meta.json");
    meta << "{\"format\":\"tifxyz\",\"type\":\"seg\","
            "\"scale\":[1.0,1.0],\"uuid\":\""
         << id
         << "\",\"bbox\":[[0.0,0.0," << z
         << "],[" << 4.f * (cols - 1) << ",28.0," << z << "]]}\n";
    REQUIRE(meta.good());
}

std::string sh(const std::string& value)
{
    std::string quoted = "'";
    for (char c : value) {
        quoted += c == '\'' ? "'\\''" : std::string(1, c);
    }
    return quoted + "'";
}

int run_cli(const std::vector<std::string>& args)
{
    std::string command = sh(VC_TIFXYZ_CONSENSUS_BIN);
    for (const std::string& arg : args) {
        command += " " + sh(arg);
    }
    command += " >/dev/null 2>&1";
    const int status = std::system(command.c_str());
    REQUIRE(status != -1);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

struct TempDir {
    fs::path path;
    TempDir()
    {
        path = fs::temp_directory_path() /
               ("vc_tifxyz_consensus_cli_" + std::to_string(::getpid()));
        fs::create_directories(path);
    }
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// reference at z=10; runs: identical, 1.5 vx above, 10 vx above (another sheet), and an identical
// run that only covers the first four columns.
struct Fixture {
    TempDir tmp;
    fs::path ref = tmp.path / "ref";
    std::vector<std::string> others;
    Fixture()
    {
        write_tifxyz(ref, "ref", 10.f);
        write_tifxyz(tmp.path / "same", "same", 10.f);
        write_tifxyz(tmp.path / "near", "near", 11.5f);
        write_tifxyz(tmp.path / "other_sheet", "other_sheet", 20.f);
        write_tifxyz(tmp.path / "half", "half", 10.f, 4);
        for (const char* name : {"same", "near", "other_sheet", "half"}) {
            others.push_back((tmp.path / name).string());
        }
    }
    std::vector<std::string> args(std::vector<std::string> extra = {}) const
    {
        std::vector<std::string> a{ref.string()};
        a.insert(a.end(), others.begin(), others.end());
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    }
};

} // namespace

TEST_CASE("consensus channel counts the runs within tolerance of each point")
{
    Fixture f;
    REQUIRE(run_cli(f.args()) == 0);

    const cv::Mat agree = cv::imread((f.ref / "consensus.tif").string(), cv::IMREAD_UNCHANGED);
    REQUIRE(agree.type() == CV_8UC1);
    REQUIRE(agree.rows == 8);
    REQUIRE(agree.cols == 8);
    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            // same + near everywhere, half only where it reaches (x <= 12, so columns 0..3)
            CHECK(agree.at<uint8_t>(v, u) == (u < 4 ? 3 : 2));
        }
    }
}

TEST_CASE("tolerance and channel name are configurable")
{
    Fixture f;
    REQUIRE(run_cli(f.args({"--tolerance", "1", "--name", "agree"})) == 0);
    CHECK_FALSE(fs::exists(f.ref / "consensus.tif"));

    const cv::Mat agree = cv::imread((f.ref / "agree.tif").string(), cv::IMREAD_UNCHANGED);
    REQUIRE(agree.type() == CV_8UC1);
    // the run 1.5 vx away no longer counts
    CHECK(agree.at<uint8_t>(0, 0) == 2);
    CHECK(agree.at<uint8_t>(0, 7) == 1);
}

TEST_CASE("min-agree writes a filtered copy and leaves the reference alone")
{
    Fixture f;
    const fs::path out = f.tmp.path / "filtered";
    REQUIRE(run_cli(f.args({"--min-agree", "3", "--output", out.string()})) == 0);

    CHECK_FALSE(fs::exists(f.ref / "consensus.tif"));
    const cv::Mat refX = cv::imread((f.ref / "x.tif").string(), cv::IMREAD_UNCHANGED);
    CHECK(cv::countNonZero(refX == -1.f) == 0);

    const cv::Mat x = cv::imread((out / "x.tif").string(), cv::IMREAD_UNCHANGED);
    REQUIRE(x.type() == CV_32FC1);
    int kept = 0;
    for (int v = 0; v < x.rows; ++v) {
        for (int u = 0; u < x.cols; ++u) {
            if (x.at<float>(v, u) != -1.f) {
                ++kept;
                CHECK(x.at<float>(v, u) <= 12.f);
            }
        }
    }
    CHECK(kept == 32);
    const cv::Mat agree = cv::imread((out / "consensus.tif").string(), cv::IMREAD_UNCHANGED);
    CHECK(agree.type() == CV_8UC1);
}

TEST_CASE("bad arguments are rejected")
{
    Fixture f;
    CHECK(run_cli({f.ref.string()}) != 0);
    CHECK(run_cli({f.ref.string(), f.ref.string()}) != 0);
    CHECK(run_cli(f.args({"--min-agree", "3"})) != 0);
    CHECK(run_cli(f.args({"--output", (f.tmp.path / "o").string()})) != 0);
    CHECK(run_cli(f.args({"--min-agree", "5", "--output", (f.tmp.path / "o").string()})) != 0);
    CHECK(run_cli(f.args({"--tolerance", "0"})) != 0);
    CHECK_FALSE(fs::exists(f.ref / "consensus.tif"));
}

#endif
