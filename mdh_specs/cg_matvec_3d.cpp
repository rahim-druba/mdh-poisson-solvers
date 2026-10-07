#include "md_hom_generator.hpp"

/**
 * L1: i index, L2: j index, L3: k index
 * R dims: none (pure stencil, no reduction)
 *
 * Matrix-free matvec Ap = A*p for the CG solver's 3D 7-point Poisson
 * stencil (diagonal 6, six face-neighbors -1), the direct 3D extension of
 * ../../cg_matvec/spec/cg_matvec.cpp's 2D 5-point cross.
 * A is never stored -- only the stencil coefficients are baked into f().
 *
 * Neighborhood is nested 3 levels deep (one per dimension): the outer N()
 * selects the L1 (i) layer (-1/0/+1), the middle N() the L2 (j) row within
 * that layer, the inner N() the L3 (k) point within that row. Only the
 * face-center of each off-layer/off-row slot is a 1 (or the true origin's
 * 2); everything else is 0 -- i.e. face neighbors only, no diagonals/edges.
 *
 * oob::ZERO matches the Dirichlet interior formulation used in
 * kernel_sparse_3d.cu: a boundary-adjacent cell has no matrix entry for a
 * missing neighbor, equivalent to treating that neighbor's p-value as 0
 * (its contribution moved to the right-hand side b during construction).
 */
int main() {
    auto P = md_hom::input_stencil_buffer(
        "P",
        {md_hom::L(1), md_hom::L(2), md_hom::L(3)},
        md_hom::N(
            md_hom::N(md_hom::N(0,0,0), md_hom::N(0,1,0), md_hom::N(0,0,0)),   // i-1 layer: only (j=0,k=0) face neighbor
            md_hom::N(md_hom::N(0,1,0), md_hom::N(1,2,1), md_hom::N(0,1,0)),   // i=0 layer: the old 2D 5-point cross
            md_hom::N(md_hom::N(0,0,0), md_hom::N(0,1,0), md_hom::N(0,0,0))    // i+1 layer: only (j=0,k=0) face neighbor
        ),
        md_hom::oob::ZERO
    );

    auto result = md_hom::result_buffer("AP", {md_hom::L(1), md_hom::L(2), md_hom::L(3)});

    auto f = md_hom::scalar_function(
        "return 6.0f * P_val - (P_val_l1_m1 + P_val_l1_p1 + P_val_l2_m1 + P_val_l2_p1 + P_val_l3_m1 + P_val_l3_p1);"
    );
    auto g = md_hom::scalar_function("return res;");

    auto md_hom_cg_matvec_3d = md_hom::md_hom<3, 0>(
        "cg_matvec_3d",
        md_hom::inputs(P),
        f, g,
        result,
        false, false
    );

    auto generator = md_hom::generator::cuda_generator(md_hom_cg_matvec_3d);
    std::ofstream kernel_file;

    kernel_file.open("cg_matvec_3d_1.cu", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_1();
    kernel_file.close();

    kernel_file.open("cg_matvec_3d_2.cu", std::fstream::out | std::fstream::trunc);
    kernel_file << generator.kernel_2();
    kernel_file.close();
}
