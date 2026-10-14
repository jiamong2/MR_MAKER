// rbshift — Rubber Band Library(R3 "finer" 엔진)로 길이를 유지한 채 음높이만 바꾸는 작은 명령행 도구.
//
// MR MAKER 가 별도 프로세스로 실행합니다 (FFmpeg 와 같은 방식).
//   입력: stdin 으로 float32 리틀엔디언 인터리브 PCM (전체)
//   출력: stdout 으로 같은 길이·같은 채널의 float32 PCM
//
// 사용: rbshift --rate 44100 --channels 2 --semitones -3 [--formant] [--engine finer|faster]
//       rbshift --version
//
// 라이선스: 이 도구는 Rubber Band Library(GPL-2.0-or-later)를 포함하므로 GPL-2.0-or-later 로 배포됩니다.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "rubberband/RubberBandStretcher.h"

using RubberBand::RubberBandStretcher;

static int usage() {
    std::fprintf(stderr, "usage: rbshift --rate HZ --channels N --semitones S [--formant] [--engine finer|faster]\n");
    return 2;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    double rate = 0, semitones = 0;
    int channels = 0;
    bool formant = false, finer = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--version") {
            std::printf("rbshift 1.0 (Rubber Band Library %s)\n", RUBBERBAND_VERSION);
            return 0;
        } else if (a == "--rate") rate = std::atof(next("--rate"));
        else if (a == "--channels") channels = std::atoi(next("--channels"));
        else if (a == "--semitones") semitones = std::atof(next("--semitones"));
        else if (a == "--formant") formant = true;
        else if (a == "--engine") finer = std::string(next("--engine")) != "faster";
        else return usage();
    }
    if (rate < 8000 || rate > 768000 || channels < 1 || channels > 16 || std::fabs(semitones) > 48) return usage();

    // 입력 전체 읽기
    std::vector<float> in;
    {
        std::vector<char> buf(1 << 20);
        std::vector<char> bytes;
        size_t got;
        while ((got = std::fread(buf.data(), 1, buf.size(), stdin)) > 0) bytes.insert(bytes.end(), buf.begin(), buf.begin() + got);
        size_t frames = bytes.size() / (sizeof(float) * channels);
        in.resize(frames * channels);
        if (frames) std::memcpy(in.data(), bytes.data(), frames * channels * sizeof(float));
    }
    const size_t n = in.size() / channels;
    if (n == 0) return 0;

    RubberBandStretcher::Options opts = RubberBandStretcher::OptionProcessOffline |
                                         RubberBandStretcher::OptionChannelsTogether |
                                         RubberBandStretcher::OptionPitchHighQuality |
                                         RubberBandStretcher::OptionThreadingNever |
                                         (finer ? RubberBandStretcher::OptionEngineFiner : RubberBandStretcher::OptionEngineFaster) |
                                         (formant ? RubberBandStretcher::OptionFormantPreserved : RubberBandStretcher::OptionFormantShifted);
    RubberBandStretcher rb((size_t)rate, (size_t)channels, opts, 1.0, std::pow(2.0, semitones / 12.0));
    rb.setDebugLevel(0);
    const size_t block = 16384;
    rb.setExpectedInputDuration(n);
    rb.setMaxProcessSize(block);

    std::vector<std::vector<float>> planar(channels, std::vector<float>(block));
    std::vector<float *> ptrs(channels);
    auto fill = [&](size_t start, size_t count) {
        for (int c = 0; c < channels; ++c) {
            for (size_t i = 0; i < count; ++i) planar[c][i] = in[(start + i) * channels + c];
            ptrs[c] = planar[c].data();
        }
    };
    // 1) study (오프라인 모드: 전체를 먼저 분석)
    for (size_t s = 0; s < n; s += block) {
        size_t cnt = std::min(block, n - s);
        fill(s, cnt);
        rb.study(ptrs.data(), cnt, s + cnt >= n);
    }
    // 2) process + retrieve
    std::vector<std::vector<float>> outPlanar(channels);
    std::vector<std::vector<float>> tmp(channels);
    std::vector<float *> optrs(channels);
    auto drain = [&]() {
        int avail;
        while ((avail = rb.available()) > 0) {
            for (int c = 0; c < channels; ++c) { tmp[c].resize(avail); optrs[c] = tmp[c].data(); }
            size_t got = rb.retrieve(optrs.data(), (size_t)avail);
            for (int c = 0; c < channels; ++c) outPlanar[c].insert(outPlanar[c].end(), tmp[c].begin(), tmp[c].begin() + got);
        }
        return avail;
    };
    for (size_t s = 0; s < n; s += block) {
        size_t cnt = std::min(block, n - s);
        fill(s, cnt);
        rb.process(ptrs.data(), cnt, s + cnt >= n);
        drain();
    }
    while (drain() >= 0) { /* available() 가 -1 이 될 때까지 (처리 완료) */ }

    // 3) 정확히 같은 길이로 출력
    std::vector<float> out(n * channels, 0.f);
    for (int c = 0; c < channels; ++c) {
        size_t m = std::min(n, outPlanar[c].size());
        for (size_t i = 0; i < m; ++i) out[i * channels + c] = outPlanar[c][i];
    }
    size_t written = std::fwrite(out.data(), sizeof(float), out.size(), stdout);
    std::fflush(stdout);
    return written == out.size() ? 0 : 3;
}
