/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/epilogue_quant_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination_dequant_bias_activation_quant.h"
#include "cutlass/gemm/kernel/gemm_quant_with_epilogue.h"
#include "cutlass/gemm/kernel/default_gemm_quant_with_broadcast.h"

#include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "testbed_gemm_with_broadcast.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Gemm>
bool test_run(
    cutlass::gemm::GemmUniversalMode mode,
    cutlass::gemm::GemmCoord problem_size,
    int batch_count = 1,
    typename Gemm::EpilogueOutputOp::ElementCompute alpha = typename Gemm::EpilogueOutputOp::ElementCompute(1),
    typename Gemm::EpilogueOutputOp::ElementCompute beta = typename Gemm::EpilogueOutputOp::ElementCompute(0)) {
    cudaStreamSynchronize(nullptr);
    cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;          // Input A
    cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;          // Input B
    cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;          // Input C
    cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementCompute, typename Gemm::LayoutC>
        tensor_Bias;   // Input Broadcast
    cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementCompute, typename Gemm::LayoutC>
        tensor_Dequant;   // Input Broadcast
    cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementCompute, typename Gemm::LayoutC>
        tensor_Quant;   // Input Broadcast

    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_Bias.resize({problem_size.m(), 1});
    tensor_Dequant.resize({problem_size.m(), 1});
    tensor_Quant.resize({problem_size.m(), 1});
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_Bias.sync_device();
    tensor_Dequant.sync_device();
    tensor_Quant.sync_device();
    typename Gemm::Arguments arguments{
        mode,
        problem_size,
        batch_count,
        {alpha, beta},
        tensor_A.device_data(),
        tensor_B.device_data(),
        tensor_C.device_data(),
        tensor_C.device_data(),
        tensor_Bias.device_data(),
        tensor_Dequant.device_data(),
        tensor_Quant.device_data(),
        problem_size.m() * problem_size.k(),
        problem_size.n() * problem_size.k(),
        problem_size.m() * problem_size.n(),
        problem_size.m() * problem_size.n(),
        problem_size.m(),
        problem_size.m(),
        problem_size.m(),
        tensor_A.layout().stride(0),
        tensor_B.layout().stride(0),
        tensor_C.layout().stride(0),
        tensor_C.layout().stride(0)
    };
    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);
    return true;
}

TEST(SM80_Device_GemmWithBroadcast_i8n_i8t_i8n_tensor_op_i32_dequant_bias, 128x128_64x5_64x64x64_16x8x32) {
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationInt8DequantBiasActivationQuant<
        cutlass::bfloat16_t,
        8,
        cutlass::divides,
        cutlass::epilogue::thread::GELU_taylor,
        cutlass::multiplies
    >;

    using GemmKernel =
        typename cutlass::gemm::kernel::DefaultGemmQuantWithBroadcast<
            int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,    // transposed B operand
            int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,    // transposed A operand
            int8_t, cutlass::layout::RowMajor,
            int32_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
            EpilogueOutputOp,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
            5,
            cutlass::arch::OpMultiplyAddSaturate
        >::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    int M_problems[] = {8 * 2, 136 * 2, 264 * 2, 520 * 2};
    int N_problems[] = {8 * 2, 136 * 2, 264 * 2, 520 * 2};
    int K_problems[] = {8 * 2, 136 * 2, 264 * 2, 520 * 2};
    double alpha_problems[] = {1.25, 2.25};
    double beta_problems[] = {0, 1, 2.0};

    for (int M : M_problems) {
        for (int N : N_problems) {
            for (int K : K_problems) {
                for (double alpha : alpha_problems) {
                    for (double beta : beta_problems) {
                        test_run<Gemm>(
                            cutlass::gemm::GemmUniversalMode::kGemm,
                            {M, N, K},
                            1,
                            cutlass::from_real<Gemm::EpilogueOutputOp::ElementCompute>(alpha),
                            cutlass::from_real<Gemm::EpilogueOutputOp::ElementCompute>(beta)
                        );
                    }
                }
            }
        }
    }
    Gemm::Arguments args{};
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
