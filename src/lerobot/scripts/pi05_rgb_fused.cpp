#include <cstddef>
#include <cstdint>

// Layout/conversion only: levels are the caller's exact Torch float32 /255 values.
extern "C" void rgb_u8_to_chw_f32(const std::uint8_t* source, float* destination,
                                  std::size_t pixels, const float* levels) noexcept {
    float* red = destination;
    float* green = destination + pixels;
    float* blue = destination + 2 * pixels;
    for (std::size_t i = 0; i < pixels; ++i) {
        red[i] = levels[source[3 * i]];
        green[i] = levels[source[3 * i + 1]];
        blue[i] = levels[source[3 * i + 2]];
    }
}
