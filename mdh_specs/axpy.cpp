#include "md_hom_generator.hpp"
// vector update  OUT[i] = A[i] + s * B[i]  (x+alpha p, r-alpha q, r+beta p): a pure map, one L dimension, one scalar input
int main() {
    auto A = md_hom::input_buffer("A", {md_hom::L(1)});
    auto B = md_hom::input_buffer("B", {md_hom::L(1)});
    auto s = md_hom::input_scalar("s");
    auto result = md_hom::result_buffer("OUT", {md_hom::L(1)});
    auto f = md_hom::scalar_function("return A_val + s_val * B_val;");
    auto g = md_hom::scalar_function("return res;");
    auto md_hom_axpy = md_hom::md_hom<1, 0>("axpy", md_hom::inputs(A, B, s), f, g, result, false, false);
    auto generator = md_hom::generator::cuda_generator(md_hom_axpy);
    std::ofstream k; k.open("axpy_1.cu", std::fstream::out | std::fstream::trunc); k << generator.kernel_1(); k.close();
    k.open("axpy_2.cu", std::fstream::out | std::fstream::trunc); k << generator.kernel_2(); k.close();
}
