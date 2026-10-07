#include "md_hom_generator.hpp"
// dot product  s = sum_k U[k] * V[k]   : 0 independent dims besides a unit L dim, one reduction dim (like matvec_dense with a vector instead of a matrix)
int main() {
    auto U = md_hom::input_buffer("U", {md_hom::R(1)});
    auto V = md_hom::input_buffer("V", {md_hom::R(1)});
    auto result = md_hom::result_buffer("S", {md_hom::L(1)});
    auto f = md_hom::scalar_function("return U_val * V_val;");
    auto g = md_hom::scalar_function("return res;");
    auto md_hom_dot = md_hom::md_hom<1, 1>("dot", md_hom::inputs(U, V), f, g, result, true, true);
    auto generator = md_hom::generator::cuda_generator(md_hom_dot);
    std::ofstream k; k.open("dot_1.cu", std::fstream::out | std::fstream::trunc); k << generator.kernel_1(); k.close();
    k.open("dot_2.cu", std::fstream::out | std::fstream::trunc); k << generator.kernel_2(); k.close();
}
