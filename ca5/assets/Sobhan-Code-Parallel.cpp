
#include <iostream>
#include <atomic>
#include <vector>
#include <cstdlib>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_group.h>
#include <tbb/spin_mutex.h>
#include <tbb/tick_count.h>
#include <tbb/info.h>

using namespace std;

int rows, cols;
atomic<int> totalPaths{0};

tbb::spin_mutex outputMutex;

int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

void createMaze(int** maze, int** visited)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, rows),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i)
                for (int j = 0; j < cols; ++j) {
                    maze[i][j] = 1;
                    visited[i][j] = 0;
                }
        });
}

void addObstacles(int** maze)
{
    if (rows > 2 && cols > 2) {
        maze[1][1] = 0;
        maze[rows - 2][cols - 2] = 0;
    }
    if (rows > 3 && cols > 3) {
        maze[2][1] = 0;
        maze[1][cols - 2] = 0;
    }
}

void displayPath(int** maze, int** visited)
{
    tbb::spin_mutex::scoped_lock lock(outputMutex);

    cout << "Path found:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (visited[i][j])      cout << " * ";
            else if (maze[i][j]==0) cout << " # ";
            else                    cout << " . ";
        }
        cout << '\n';
    }
    cout << '\n';
}

bool isValidMove(int x, int y, int** maze, int** visited)
{
    return x >= 0 && x < rows &&
           y >= 0 && y < cols &&
           maze[x][y] == 1 &&
           visited[x][y] == 0;
}

int** copyVisited(int** src)
{
    int** dst = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        dst[i] = new int[cols];
        for (int j = 0; j < cols; ++j)
            dst[i][j] = src[i][j];
    }
    return dst;
}

void freeVisited(int** v)
{
    for (int i = 0; i < rows; ++i) delete[] v[i];
    delete[] v;
}

void findAllPathsParallel(int x, int y, int ex, int ey,
                          int** maze, int** visited, int depth)
{
    if (x == ex && y == ey) {
        visited[x][y] = 1;
        displayPath(maze, visited);
        ++totalPaths;
        visited[x][y] = 0;
        return;
    }

    visited[x][y] = 1;

    vector<pair<int,int>> moves;
    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (isValidMove(nx, ny, maze, visited))
            moves.push_back({nx, ny});
    }

    if (depth < 3 && moves.size() > 1) {
        tbb::task_group tg;
        for (const auto& mv : moves) {
            int nx = mv.first;
            int ny = mv.second;
            int** vcopy = copyVisited(visited);

            tg.run([=] {
                findAllPathsParallel(nx, ny, ex, ey, maze, vcopy, depth + 1);
                freeVisited(vcopy);
            });
        }
        tg.wait();
    } else {
        for (const auto& mv : moves) {
            int nx = mv.first;
            int ny = mv.second;
            int** vcopy = copyVisited(visited);

            findAllPathsParallel(nx, ny, ex, ey, maze, vcopy, depth + 1);
            freeVisited(vcopy);
        }
    }

    visited[x][y] = 0;
}

void findAllPathsSequential(int x, int y, int ex, int ey,
                            int** maze, int** visited, int& count)
{
    if (x == ex && y == ey) {
        ++count;
        return;
    }

    visited[x][y] = 1;

    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (isValidMove(nx, ny, maze, visited))
            findAllPathsSequential(nx, ny, ex, ey, maze, visited, count);
    }

    visited[x][y] = 0;
}

int main(int argc, char* argv[])
{
    rows = (argc > 1) ? atoi(argv[1]) : 5;
    cols = (argc > 2) ? atoi(argv[2]) : 5;

    cout << "Maze size: " << rows << "x" << cols << '\n';
    cout << "TBB concurrency: "
         << tbb::info::default_concurrency() << '\n';

    int** maze = new int*[rows];
    int** visited = new int*[rows];
    for (int i = 0; i < rows; ++i) {
        maze[i] = new int[cols];
        visited[i] = new int[cols];
    }

    createMaze(maze, visited);
    addObstacles(maze);

    if (maze[0][0] == 0 || maze[rows-1][cols-1] == 0) {
        cerr << "Start or end blocked\n";
        return 1;
    }

    cout << "\n[Sequential]\n";
    int seqCount = 0;
    tbb::tick_count s0 = tbb::tick_count::now();
    findAllPathsSequential(0, 0, rows-1, cols-1, maze, visited, seqCount);
    double seqTime = (tbb::tick_count::now() - s0).seconds() * 1000;

    cout << "Paths: " << seqCount << '\n';
    cout << "Time: " << seqTime << " ms\n";

    cout << "\n[Parallel]\n";
    totalPaths = 0;
    tbb::tick_count p0 = tbb::tick_count::now();
    findAllPathsParallel(0, 0, rows-1, cols-1, maze, visited, 0);
    double parTime = (tbb::tick_count::now() - p0).seconds() * 1000;

    cout << "Paths: " << totalPaths << '\n';
    cout << "Time: " << parTime << " ms\n";

    cout << "\nSpeedup: " << seqTime / parTime << "x\n";

    for (int i = 0; i < rows; ++i) {
        delete[] maze[i];
        delete[] visited[i];
    }
    delete[] maze;
    delete[] visited;

    return 0;
}
