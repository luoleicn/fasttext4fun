
set(DEP_LIBS gtest gtest_main pthread fasttext)

add_executable(test_mat test_mat.cpp)
TARGET_LINK_LIBRARIES(test_mat
    ${DEP_LIBS}
)

add_executable(test_data_loader test_data_loader.cpp)
TARGET_LINK_LIBRARIES(test_data_loader
    ${DEP_LIBS}
)

add_executable(test_model test_model.cpp)
TARGET_LINK_LIBRARIES(test_model
    ${DEP_LIBS}
)

