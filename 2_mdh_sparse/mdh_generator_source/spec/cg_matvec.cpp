#include "md_hom_generator.hpp"

/**
 * L1: row index (i)
 * L2: col index (j)
 * R dims: none (pure stencil, no reduction)
 *
 * Matrix-free matvec Ap = A*p for the CG solver's 2D 5-point Poisson stencil
 * (diagonal 4, four neighbors -1, matching kernel_sparse.cu's CSR matrix).
 * A is never stored -- only the stencil coefficients are baked into f().
 *
 * Neighborhood N(N(0,1,0), N(1,2,1), N(0,1,0)) is the exact 5-point cross:
 * top(-1,0), left(0,-1), center(0,0), right(0,+1), bottom(+1,0).
 *
 * oob::ZERO matches the Dirichlet interior formulation used in kernel_sparse.cu:
 * a boundary-adjacent row has no matrix entry for the missing neighbor, which is
 * equivalent to treating that neighbor's p-value as 0 (its contribution moved to
 * the right-hand side b during matrix construction).
 */
int main() {
    auto P = md_hom::input_stencil_buffer(
        "P",
        {md_hom::L(1), md_hom::L(2)},
        md_hom::N(md_hom::N(0,1,0), md_hom::N(1,2,1), md_hom::N(0,1,0)),
        md_hom::oob::ZERO
    );

    auto result = md_hom::result_buffer("AP", {md_hom::L(1), md_hom::L(2)});

    auto f = md_hom::scalar_function(
        "return 4.0f * P_val - (P_val_l1_m1 + P_val_l1_p1 + P_val_l2_m1 + P_val_l2_p1);"
    );
    auto g = md_hom::scalar_function("return res;");

    auto md_hom_cg_matvec = md_hom::md_hom<2, 0>(
        "cg_matvec",
        md_hom::inputs(P),
        f, g,
        result,
        false, false
    );

    auto generator = md_hom::generator::cuda_generator(md_hom_cg_matvec);
    std::ofstream kernel_file;

    kernel_file.open("cg_matvec_1.cu", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_1();
    kernel_file.close();

    kernel_file.open("cg_matvec_2.cu", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_2();
    kernel_file.close();
}
