// Sequential Maze Solver - Find all paths from start to end
// Reference: Backtracking maze pathfinding algorithm
#include <iostream>
#include <omp.h>
using namespace std;

int rows, cols;

int serialTotalPaths = 0;
int parallelTotalPaths = 0;

int maxDepth; // Example threshold for depth

int dx[] = { -1, 1, 0, 0 };
int dy[] = { 0, 0, -1, 1 };

void createMaze(int** maze, int** visited)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            maze[i][j] = 1;
            visited[i][j] = 0;
        }
    }
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

void displayMaze(int** maze, int** visited)
{
    cout << "Path found:" << endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (visited[i][j] == 1)
                cout << " * ";
            else if (maze[i][j] == 0)
                cout << " # ";
            else
                cout << " . ";
        }
        cout << endl;
    }
    cout << endl;
}

bool isValidMove(int x, int y, int** maze, int** visited)
{
    if (x >= 0 && x < rows && y >= 0 && y < cols) {
        if (maze[x][y] == 1 && visited[x][y] == 0) {
            return true;
        }
    }
    return false;
}

void copyVisited(int** source, int** dest)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i][j] = source[i][j];
        }
    }
}

void findAllPaths_serial(int x, int y, int endX, int endY, int** maze, int** visited)
{
    if (x == endX && y == endY) {
        serialTotalPaths++;
        return;
    }

    visited[x][y] = 1;

    for (int dir = 0; dir < 4; dir++) {
        int newX = x + dx[dir];
        int newY = y + dy[dir];

        if (isValidMove(newX, newY, maze, visited)) {
            findAllPaths_serial(newX, newY, endX, endY, maze, visited);
        }
    }

    visited[x][y] = 0;
}

void findAllPaths_parallel(int x, int y, int endX, int endY,
    int** maze, int** visited, int depth)
{
    if (x == endX && y == endY) {
        //displayMaze(maze, visited);
        #pragma omp atomic
        parallelTotalPaths++;
        return;
    }

    visited[x][y] = 1;

    for (int dir = 0; dir < 4; dir++) {
        int newX = x + dx[dir];
        int newY = y + dy[dir];

        if (isValidMove(newX, newY, maze, visited)) {
            if (depth < maxDepth) {
                int** visitedCopy = new int* [rows];
                for (int i = 0; i < rows; i++) {
                    visitedCopy[i] = new int[cols];
                }

                copyVisited(visited, visitedCopy);

                #pragma omp task firstprivate(visitedCopy, newX, newY, depth)
                {
                    findAllPaths_parallel(newX, newY, endX, endY, maze, visitedCopy, depth + 1);
                    // free(visitedCopy);
                    for (int i = 0; i < rows; i++) {
                        delete[] visitedCopy[i];
                    }
                    delete[] visitedCopy;
                }
            }
            else {
                findAllPaths_parallel(newX, newY, endX, endY, maze, visited, depth + 1);
            }
        }
    }
    visited[x][y] = 0;
}

void resetVisited(int** visited, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            visited[i][j] = 0;
}

int main()
{
    rows = 7;
    cols = 7;
	maxDepth = 11;

    int** maze = new int* [rows];
    int** visited = new int* [rows];
    for (int i = 0; i < rows; i++) {
        maze[i] = new int[cols];
        visited[i] = new int[cols];
    }

    createMaze(maze, visited);
    addObstacles(maze);

    cout << "Finding all paths from (0,0) to ("
        << rows - 1 << "," << cols - 1 << ")" << endl;
    cout << "================================" << endl;

    double startTime = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            findAllPaths_parallel(0, 0, rows - 1, cols - 1, maze, visited, 0);
        }
    }
    double endTime = omp_get_wtime();
    double parallel_time = endTime - startTime;

    resetVisited(visited, rows, cols);

    startTime = omp_get_wtime();
    findAllPaths_serial(0, 0, rows - 1, cols - 1, maze, visited);
    endTime = omp_get_wtime();
    double serial_time = endTime - startTime;

    cout << "================================" << endl;
    cout << "serial Total paths found: " << serialTotalPaths << endl;
    cout << "serial Time: " << serial_time << endl;

    cout << "parallel Total paths found: " << parallelTotalPaths << endl;
    cout << "parallel Time: " << parallel_time << endl;
    cout << "================================" << endl;
    cout << "Speedup: " << serial_time / parallel_time << endl;

    for (int i = 0; i < rows; i++) {
        delete[] maze[i];
        delete[] visited[i];
    }
    delete[] maze;
    delete[] visited;

    return 0;
}