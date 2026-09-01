add_subdirectory(01-execution-barriers)
add_subdirectory(02-memory-ordering)
add_subdirectory(03-async-pipelines)
add_subdirectory(04-tensor-synchronization)
add_subdirectory(05-collectives-and-atomics)

add_custom_target(part13_sync_inspect
  DEPENDS
    part13_execution_barriers_inspect
    part13_memory_order_inspect
    part13_async_pipelines_inspect
    part13_tensor_synchronization_inspect
    part13_collectives_atomics_inspect
)
