// Sequential Maze Solver - Find all paths from start to end
// Reference: Backtracking maze pathfinding algorithm
#include <iostream>
using namespace std;

int rows, cols;
int totalPaths = 0;

int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

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
        maze[rows-2][cols-2] = 0;
    }
    if (rows > 3 && cols > 3) {
        maze[2][1] = 0;
        maze[1][cols-2] = 0;
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

void findAllPaths(int x, int y, int endX, int endY, 
                  int** maze, int** visited)
{
    if (x == endX && y == endY) {
        visited[x][y] = 1;
        displayMaze(maze, visited);
        totalPaths++;
        visited[x][y] = 0;
        return;
    }
    
    visited[x][y] = 1;
    
    for (int dir = 0; dir < 4; dir++) {
        int newX = x + dx[dir];
        int newY = y + dy[dir];
        
        if (isValidMove(newX, newY, maze, visited)) {
            int** newVisited = new int*[rows];
            for (int i = 0; i < rows; i++) {
                newVisited[i] = new int[cols];
            }
            copyVisited(visited, newVisited);
            
            findAllPaths(newX, newY, endX, endY, maze, newVisited);
            
            for (int i = 0; i < rows; i++) {
                delete[] newVisited[i];
            }
            delete[] newVisited;
        }
    }
    
    visited[x][y] = 0;
}

int main()
{
    rows = 4;
    cols = 4;
    
    int** maze = new int*[rows];
    int** visited = new int*[rows];
    for (int i = 0; i < rows; i++) {
        maze[i] = new int[cols];
        visited[i] = new int[cols];
    }
    
    createMaze(maze, visited);
    addObstacles(maze);
    
    cout << "Finding all paths from (0,0) to (" 
         << rows-1 << "," << cols-1 << ")" << endl;
    cout << "================================" << endl;
    
    findAllPaths(0, 0, rows-1, cols-1, maze, visited);
    
    cout << "================================" << endl;
    cout << "Total paths found: " << totalPaths << endl;
    
    for (int i = 0; i < rows; i++) {
        delete[] maze[i];
        delete[] visited[i];
    }
    delete[] maze;
    delete[] visited;
    
    return 0;
}

