#include <catch2/catch_test_macros.hpp>

TEST_CASE("sanity check", "[smoke]") {
    REQUIRE(1 + 1 == 2);
}
