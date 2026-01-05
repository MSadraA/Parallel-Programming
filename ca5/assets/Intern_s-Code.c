//Game Score Manager - Buggy Code

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_NAME_LENGTH 20
#define MAX_PLAYERS 5
#define SCORE_MULTIPLIER 100

const char GAME_PREFIX[] = "GAME";

char* createPlayer(char* playerName)
{
    char* player = (char*)(malloc(sizeof(char) * MAX_NAME_LENGTH));
    if (playerName != NULL)
    {
        for (int i = 0; i < MAX_NAME_LENGTH; ++i)
        {
            srand(time(0));
            player[i] = rand() % 26 + 'A';
        }
        return player;
    }
    return NULL;
}

int* calculateScore(char* playerId)
{
    int* score = (int*)(malloc(sizeof(int)));
    int i = 0;
    if (playerId != NULL)
    {
        while (1)
        {
            if (i >= strlen(playerId))
                return score;
            else if (i < strlen(GAME_PREFIX))
                *score += GAME_PREFIX[i];
            else
                *score += playerId[i] * SCORE_MULTIPLIER;
            i++;
        }
    }
    return NULL;
}

char* getPlayerRank(int* playerScore)
{
    if (*playerScore < 0)
        return "Invalid";
    
    char* rank;
    int scoreValue = *playerScore;
    
    if (scoreValue > 5000)
    {
        rank = (char*)(malloc(sizeof(char) * 10));
        strncpy(rank, "Champion", 8);
    }
    else if (scoreValue > 3000)
    {
        rank = (char*)(malloc(sizeof(char) * 10));
        strncpy(rank, "Expert", 6);
    }
    else if (scoreValue > 1000)
    {
        rank = (char*)(malloc(sizeof(char) * 10));
        strncpy(rank, "Advanced", 8);
    }
    else
    {
        rank = (char*)(malloc(sizeof(char) * 10));
        strncpy(rank, "Beginner", 8);
    }
    
    return rank;
}

char* initGameSession(char* playerName)
{
    if (strncmp(playerName, GAME_PREFIX, strlen(GAME_PREFIX)) != 0)
        return playerName;
    
    int choice;
    printf(
        "Select game mode:\n"
        "0: Single Player\n"
        "1: Multiplayer\n"
        "2: Tournament\n"
        "3: Practice\n"
        "Enter choice: ");
    scanf("%d", &choice);
    
    playerName = (char*)(malloc(sizeof(char) * 15));
    switch (choice)
    {
        case(0):
            strncpy(playerName, "SINGLE", 6);
            break;
        case(1):
            strncpy(playerName, "MULTI", 5);
            break;
        case(2):
            strncpy(playerName, "TOURNAMENT", 10);
            break;
        case(3):
            strncpy(playerName, "PRACTICE", 8);
            break;
    }
    return playerName;
}

void createLeaderboard(char** players, int* scores, int count)
{
    printf("\n=== LEADERBOARD ===\n");
    for (int i = 0; i < count; i++)
    {
        printf("Player %d: %s - Score: %d\n", i+1, players[i], scores[i]);
    }
}

int main(int argc, char* argv[])
{
    char* player = createPlayer(argv[1]);
    printf("Player created: %s\n", player);
    free(player);
    
    int* score = calculateScore(player);
    printf("Player score: %d\n", *score);
    
    char* rank = getPlayerRank(score);
    printf("Player rank: %s\n", rank);
    
    char* session = initGameSession(rank);
    printf("Game session: %s\n", session);
    
    free(score);
    free(rank);
    free(session);
    
    return EXIT_SUCCESS;
}
