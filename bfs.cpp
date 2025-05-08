#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cstdlib>
using namespace std;    

int visited[1000], visit[1000];
int qu[1000], front = 0, rear = 0;
int stk[1000], top = -1;

void bfs_sequential(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    cout << "Sequential BFS: ";
    visited[start] = 1;
    cout << start << " ";
    qu[rear++] = start;

    while (front < rear) {
        int v = qu[front++];
        for (int j = 0; j < n; j++) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                visit[j] = visited[j] = 1;
                qu[rear++] = j;
                cout << j << " ";
            }
        }
    }
    cout << endl;
}

void bfs_parallel(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    cout << "Parallel BFS: ";
    visited[start] = 1;
    cout << start << " ";
    qu[rear++] = start;

    while (front < rear) {
        int v = qu[front++];

        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = visited[j] = 1;
                    qu[rear++] = j;
                    cout << j << " ";
                }
            }
        }
    }
    cout << endl;
}

void dfs_sequential(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    cout << "Sequential DFS: ";
    visited[start] = 1;
    cout << start << " ";
    stk[++top] = start;

    while (top >= 0) {
        int v = stk[top--];
        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                visit[j] = visited[j] = 1;
                stk[++top] = j;
                cout << j << " ";
            }
        }
    }
    cout << endl;
}

void dfs_parallel(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    cout << "Parallel DFS: ";
    visited[start] = 1;
    cout << start << " ";
    stk[++top] = start;

    while (top >= 0) {
        int v = stk[top--];

        #pragma omp parallel for
        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = visited[j] = 1;
                    stk[++top] = j;
                    cout << j << " ";
                }
            }
        }
    }
    cout << endl;
}

int main() {
    const int n = 200;
    int cost[1000][1000];

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            cost[i][j] = cost[j][i] = (i != j) ? rand() % 2 : 0;

    int start;

    cout << "Enter starting vertex for BFS: ";
    cin >> start;
    auto t1 = chrono::high_resolution_clock::now();
    bfs_sequential(cost, n, start);
    auto t2 = chrono::high_resolution_clock::now();
    bfs_parallel(cost, n, start);
    auto t3 = chrono::high_resolution_clock::now();

    cout << "\nEnter starting vertex for DFS: ";
    cin >> start;
    auto t4 = chrono::high_resolution_clock::now();
    dfs_sequential(cost, n, start);
    auto t5 = chrono::high_resolution_clock::now();
    dfs_parallel(cost, n, start);
    auto t6 = chrono::high_resolution_clock::now();

    auto bfs_seq_time = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    auto bfs_par_time = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();
    auto dfs_seq_time = chrono::duration_cast<chrono::microseconds>(t5 - t4).count();
    auto dfs_par_time = chrono::duration_cast<chrono::microseconds>(t6 - t5).count();

    cout << "\n--- Speedup Summary ---\n";
    cout << "BFS Seq Time: " << bfs_seq_time << " µs\n";
    cout << "BFS Par Time: " << bfs_par_time << " µs\n";
    cout << "BFS Speedup:  " << (float)bfs_seq_time / bfs_par_time << "x\n";

    cout << "DFS Seq Time: " << dfs_seq_time << " µs\n";
    cout << "DFS Par Time: " << dfs_par_time << " µs\n";
    cout << "DFS Speedup:  " << (float)dfs_seq_time / dfs_par_time << "x\n";

    return 0;
}

//g++ -fopenmp main.cpp -o main.out