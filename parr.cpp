#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

'void displayArray(int nums[], int length) {
    cout << "Nums: [";
    for (int i = 0; i < length; i++) {
        cout << nums[i];
        if (i != length - 1)
            cout << ", ";
    }
    cout << "]" << endl;
}'

int minSeq(int nums[], int length) {
    int minValue = nums[0];
    for (int i = 0; i < length; i++) {
        if (nums[i] < minValue)
            minValue = nums[i];
    }
    return minValue;
}

int maxSeq(int nums[], int length) {
    int maxValue = nums[0];
    for (int i = 0; i < length; i++) {
        if (nums[i] > maxValue)
            maxValue = nums[i];
    }
    return maxValue;
}

int sumSeq(int nums[], int length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }
    return sum;
}

float avgSeq(int nums[], int length) {
    return static_cast<float>(sumSeq(nums, length)) / length;
}

int minParallel(int nums[], int length) {
    int minValue = nums[0];
#pragma omp parallel for reduction(min : minValue)
    for (int i = 0; i < length; i++) {
        if (nums[i] < minValue)
            minValue = nums[i];
    }
    return minValue;
}

int maxParallel(int nums[], int length) {
    int maxValue = nums[0];
#pragma omp parallel for reduction(max : maxValue)
    for (int i = 0; i < length; i++) {
        if (nums[i] > maxValue)
            maxValue = nums[i];
    }
    return maxValue;
}

int sumParallel(int nums[], int length) {
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }
    return sum;
}

float avgParallel(int nums[], int length) {
    return static_cast<float>(sumParallel(nums, length)) / length;
}

int main() {
    int nums[] = {4, 6, 3, 2, 6, 7, 9, 2, 1, 6, 5};
    int length = sizeof(nums) / sizeof(int);

    //displayArray(nums, length);

    auto t1 = high_resolution_clock::now();
    int min_s = minSeq(nums, length);
    auto t2 = high_resolution_clock::now();
    int max_s = maxSeq(nums, length);
    auto t3 = high_resolution_clock::now();
    int sum_s = sumSeq(nums, length);
    auto t4 = high_resolution_clock::now();
    float avg_s = avgSeq(nums, length);
    auto t5 = high_resolution_clock::now();

    auto min_s_time = duration_cast<microseconds>(t2 - t1).count();
    auto max_s_time = duration_cast<microseconds>(t3 - t2).count();
    auto sum_s_time = duration_cast<microseconds>(t4 - t3).count();
    auto avg_s_time = duration_cast<microseconds>(t5 - t4).count();

    auto p1 = high_resolution_clock::now();
    int min_p = minParallel(nums, length);
    auto p2 = high_resolution_clock::now();
    int max_p = maxParallel(nums, length);
    auto p3 = high_resolution_clock::now();
    int sum_p = sumParallel(nums, length);
    auto p4 = high_resolution_clock::now();
    float avg_p = avgParallel(nums, length);
    auto p5 = high_resolution_clock::now();

    auto min_p_time = duration_cast<microseconds>(p2 - p1).count();
    auto max_p_time = duration_cast<microseconds>(p3 - p2).count();
    auto sum_p_time = duration_cast<microseconds>(p4 - p3).count();
    auto avg_p_time = duration_cast<microseconds>(p5 - p4).count();

    cout << "\nMinimum:\n";
    cout << "  Sequential Time: " << min_s_time << " microseconds\n";
    cout << "  Parallel Time: " << min_p_time << " microseconds\n";
    cout << "  Speedup: " << (float)min_s_time / min_p_time << "\n";

    cout << "\nMaximum:\n";
    cout << "  Sequential Time: " << max_s_time << " microseconds\n";
    cout << "  Parallel Time: " << max_p_time << " microseconds\n";
    cout << "  Speedup: " << (float)max_s_time / max_p_time << "\n";

    cout << "\nSum:\n";
    cout << "  Sequential Time: " << sum_s_time << " microseconds\n";
    cout << "  Parallel Time: " << sum_p_time << " microseconds\n";
    cout << "  Speedup: " << (float)sum_s_time / sum_p_time << "\n";

    cout << "\nAverage:\n";
    cout << "  Sequential Time: " << avg_s_time << " microseconds\n";
    cout << "  Parallel Time: " << avg_p_time << " microseconds\n";
    cout << "  Speedup: " << (float)avg_s_time / avg_p_time << "\n";

    return 0;
}