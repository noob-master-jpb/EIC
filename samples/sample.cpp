```cpp
#include "matmul_autotune.h"
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

namespace {
    // RAII wrapper for CUDA events
    class CudaEvent {
    public:
        CudaEvent() {
            cudaEventCreate(&event_);
        }
        ~CudaEvent() {
            if (event_) {
                cudaEventDestroy(event_);
            }
        }
        cudaEvent_t get() const { return event_; }
        // Prevent copying
        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;
    private:
        cudaEvent_t event_ = nullptr;
    };
}

int execute_autotuned_matmul(
    cudnnHandle_t handle,
    void* d_A,
    void* d_B,
    void* d_C,
    int64_t batch,
    int64_t m,
    int64_t n,
    int64_t k)
{
    namespace fe = cudnn_frontend;

    // Return error code for invalid parameters
    if (batch <= 0 || m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }

    // ==========================================
    // 1. Create the Graph and Matmul Operation
    // ==========================================
    auto graph = fe::graph::Graph();
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Create tensors with proper UIDs
    auto A_tensor = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("A")
            .set_dim({batch, m, k})
            .set_stride({m * k, k, 1})
    );

    auto B_tensor = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_name("B")
            .set_dim({batch, k, n})
            .set_stride({k * n, n, 1})
    );

    auto matmul_op = fe::graph::Matmul_attributes()
        .set_name("matmul")
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto [C_tensor, status] = graph.matmul(A_tensor, B_tensor, matmul_op);
    if (status.is_bad()) {
        std::cerr << "Matmul creation failed: " << status.get_message() << std::endl;
        return -1;
    }

    C_tensor->set_output(true).set_data_type(fe::DataType_t::HALF);

    // ==========================================
    // 2. Build the Operation Graph
    // ==========================================
    status = graph.validate();
    if (status.is_bad()) {
        std::cerr << "Graph validation failed: " << status.get_message() << std::endl;
        return -1;
    }

    status = graph.build_operation_graph(handle);
    if (status.is_bad()) {
        std::cerr << "Build operation graph failed: " << status.get_message() << std::endl;
        return -1;
    }

    // ==========================================
    // 3. Create Execution Plans
    // ==========================================
    auto heuristics = graph.get_heuristics();

    int num_plans = 0;
    status = heuristics.get_plan_count(num_plans);
    if (status.is_bad() || num_plans <= 0) {
        std::cerr << "No execution plans found" << std::endl;
        return -1;
    }

    // ==========================================
    // 4. Determine Workspace Requirements
    // ==========================================
    size_t max_workspace_size = 0;
    std::vector<fe::graph::Execution_plan> plans;
    plans.reserve(num_plans);

    for (int i = 0; i < num_plans; i++) {
        fe::graph::Execution_plan plan;
        plan.set_tag(std::to_string(i));
        status = heuristics.get_plan(i, plan);
        
        if (status.is_bad()) {
            continue;
        }
        
        size_t workspace = 0;
        status = plan.get_workspace_size(workspace);
        if (status.is_bad()) {
            continue;
        }
        
        max_workspace_size = std::max(max_workspace_size, workspace);
        plans.push_back(std::move(plan));
    }

    if (plans.empty()) {
        std::cerr << "No valid execution plans after filtering" << std::endl;
        return -1;
    }

    // Allocate workspace
    void* workspace = nullptr;
    if (max_workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, max_workspace_size);
        if (cuda_err != cudaSuccess) {
            std::cerr << "Workspace allocation failed: " << cudaGetErrorString(cuda_err) << std::endl;
            return -1;
        }
    }

    // ==========================================
    // 5. Autotuning: Benchmark Each Plan
    // ==========================================
    const int warmup_iters = 1;
    const int timed_iters = 10;
    const float min_time = std::numeric_limits<float>::max();
    int best_plan_idx = -1;
    float best_time = min_time;

    // Create variant pack for execution
    fe::graph::Variant_pack variant_pack;
    variant_pack.set_workspace(workspace);

    // Create CUDA events for timing
    CudaEvent start_event, stop_event;

    for (size_t i = 0; i < plans.size(); i++) {
        auto& plan = plans[i];
        
        size_t plan_workspace = 0;
        status = plan.get_workspace_size(plan_workspace);
        if (status.is_bad()) {
            continue;
        }

        // Build the plan
        status = plan.build(handle, plan_workspace, workspace);
        if (status.is_bad()) {
            continue;
        }

        // Setup variant pack with tensor pointers
        variant_pack.set_tensor(A_tensor->get_uid(), d_A)
                    .set_tensor(B_tensor->get_uid(), d_B)
                    .set_tensor(C_tensor->get_uid(), d_C);

        // Warmup iteration
        status = graph.execute(handle, plan, variant_pack);
        if (status.is_bad()) {
            continue;
        }
        cudaDeviceSynchronize();

        // Timed iterations
        cudaEventRecord(start_event.get());
        bool exec_success = true;
        
        for (int iter = 0; iter < timed_iters; iter++) {
            status = graph.execute(handle, plan, variant_pack);
            if (status.is_bad()) {
                exec_success = false;
                break;
            }
        }
        
        if (!exec_success) {
            continue;
        }
        
        cudaEventRecord(stop_event.get());
        cudaEventSynchronize(stop_event.get());

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_event.get(), stop_event.get());
        float avg_time = elapsed_ms / timed_iters;

        if (avg_time < best_time) {
            best_time = avg_time;
            best_plan_idx = static_cast<int>(i);
        }
    }

    // ==========================================
    // 6. Execute with Best Plan
    // ==========================================
    if (best_plan_idx < 0) {
        std::cerr << "All plans failed during autotuning" << std::endl;
        if (workspace) cudaFree(workspace);
        return -1;
    }

    // Execute final computation with the best plan
    auto& best_plan = plans[best_plan_idx];
    
    size_t best_workspace = 0;
    status = best_plan.get_workspace_size(best_workspace);
    if (status.is_bad()) {
        if (workspace) cudaFree(workspace);
        return -1;
    }
    
    status = best_plan.build(handle, best_workspace, workspace);
    if (status.is_bad()) {
        std::cerr << "Failed to build best plan: " << status.get_message() << std::endl;
        if (workspace) cudaFree(workspace);
        return -1;
    }

    fe::graph::Variant_pack final_variant_pack;
    final_variant_pack.set_workspace(workspace);
    final_variant_pack.set_tensor(A_tensor->get_uid(), d_A)
                      .set_tensor(B_tensor->get_uid(), d_B)
                      .set_tensor(C_tensor->get_uid(), d_C);

    status = graph.execute(handle, best_plan, final_variant_pack);
    if (status.is_bad()) {
        std::cerr << "Final execution failed: " << status.get_message() << std::endl;
        if (workspace) cudaFree(workspace);
        return -1;
    }

    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }

    return best_plan_idx;
}
```
