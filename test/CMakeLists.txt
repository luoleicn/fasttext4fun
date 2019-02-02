
add_executable(test_mat test_mat.cpp)
TARGET_LINK_LIBRARIES(test_mat
        gtest gtest_main pthread
        fasttext
)

