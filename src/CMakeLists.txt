
set (COMMON_SRC data_loader.cpp model.cpp optimizer/sgd_optimizer.cpp
	acc_evaluator.cpp)

if (USE_CPU)
    add_library(fasttext 
        cpu/cpu_mat.cpp 
        cpu/mat_factory.cpp
	${COMMON_SRC}
        )
    TARGET_LINK_LIBRARIES(fasttext
        )  

elseif(NOT USE_CPU)
    add_library(fasttext 
        gpu/cpu_mat.cpp 
        gpu/mat_factory.cpp
	${COMMON_SRC}
        )
    TARGET_LINK_LIBRARIES(fasttext
        )  
endif ()

add_executable(trainer train.cpp)
TARGET_LINK_LIBRARIES(trainer
	fasttext
	)

add_executable(tester test.cpp)
TARGET_LINK_LIBRARIES(tester
	fasttext
	)
