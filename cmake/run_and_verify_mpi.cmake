# ==============================================================================
# MPI Test Runner with Verification
# ==============================================================================

# Build MPI command
set(MPI_CMD ${MPIEXEC} ${MPI_NUMPROC_FLAG} ${MPI_RANKS})

if(MPI_OVERSUBSCRIBE)
  list(APPEND MPI_CMD ${MPI_OVERSUBSCRIBE})
endif()

list(APPEND MPI_CMD ${TEST_EXEC})

# Execute deck with MPI
execute_process(
  COMMAND ${MPI_CMD}
  RESULT_VARIABLE deck_result
  OUTPUT_VARIABLE deck_output
  ERROR_VARIABLE deck_error
)

if(NOT deck_result EQUAL 0)
  message(FATAL_ERROR "Deck execution failed (Exit: ${deck_result}):\nSTDOUT: ${deck_output}\nSTDERR: ${deck_error}")
endif()

# Run verification script
if(VERIFY AND EXISTS "${VERIFY_SCRIPT}")
  # FIX: VERIFY_ARGS is already a CMake list - just expand it directly
  execute_process(
    COMMAND ${VERIFY} ${VERIFY_SCRIPT} ${VERIFY_ARGS}
    RESULT_VARIABLE verify_result
    OUTPUT_VARIABLE verify_output
    ERROR_VARIABLE verify_error
  )

  message("${verify_output}")

  if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "Verification failed (Exit: ${verify_result}):\nSTDERR: ${verify_error}")
  endif()
else()
  message(WARNING "Verification skipped: VERIFY='${VERIFY}', Script exists: ${EXISTS ${VERIFY_SCRIPT}}")
endif()