//Game Score Manager - Fixed code

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
	// حافظه تخصیص دهیم یکجا هم برای نال در آخر رشته درنظر می گیریم تا برای پیدا کردن آن از محدوده بافر خارج نشویم MAX_NAME_LENGTH بجای اینکه دقیقا به اندازه 
	char* player = (char*)(malloc(sizeof(char) * (MAX_NAME_LENGTH + 1)));
	if (playerName != NULL)
	{
		for (int i = 0; i < MAX_NAME_LENGTH; ++i)
		{
			//srand(time(0)); این خط باید خارج از حلقه قرار گیرد تا در هر بار اجرای برنامه, رشته های متفاوتی تولید شود و همچنین در صورت اجرای سریع برنامه, رشته های یکسان تولید نشود
			player[i] = rand() % 26 + 'A';
		}
		// رشته را نال ترمینیت می کنیم تا بتوانیم از توابع رشته ای استفاده کنیم
		player[MAX_NAME_LENGTH] = '\0';
		return player;
	}
	free(player); // جلوگیری از نشت حافظه اگر ورودی نال بود
	return NULL;
}

int* calculateScore(char* playerId)
{
	int* score = (int*)(malloc(sizeof(int)));
	// امتیاز را صفر قرار می دهیم تا بتوانیم به آن امتیاز اضافه کنیم البته این فقط خطای منطقی ایجاد می کند
	*score = 0;
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
	free(score); // جلوگیری از نشت حافظه اگر ورودی نال بود
	return NULL;
}

char* getPlayerRank(int* playerScore)
{
	// اضافه کردن شرط playerScore == NULL برای جلوگیری از کرش کردن برنامه هنگام دسترسی به پوینتر نال
	if (playerScore == NULL || *playerScore < 0) {
		// برای زمانی که امتیاز منفی است، یک رشته ثابت برمی گردانیم تا از مشکلات حافظه و آزادسازی حافظه تخصیص داده نشده جلوگیری کنیم
		char* invalidRank = (char*)malloc(10 * sizeof(char));
		strcpy(invalidRank, "Invalid");
		return invalidRank;
	}

	char* rank;
	int scoreValue = *playerScore;

	// برای همه رشته های رتبه بندی حافظه تخصیص می دهیم تا از مشکلات حافظه جلوگیری کنیم و همچنین رشته ها را نال ترمینیت می کنیم
	if (scoreValue > 5000)
	{
		rank = (char*)(malloc(sizeof(char) * 10));
		strncpy(rank, "Champion", 8);
		rank[8] = '\0';
	}
	else if (scoreValue > 3000)
	{
		rank = (char*)(malloc(sizeof(char) * 10));
		strncpy(rank, "Expert", 6);
		rank[6] = '\0';
	}
	else if (scoreValue > 1000)
	{
		rank = (char*)(malloc(sizeof(char) * 10));
		strncpy(rank, "Advanced", 8);
		rank[8] = '\0';
	}
	else
	{
		rank = (char*)(malloc(sizeof(char) * 10));
		strncpy(rank, "Beginner", 8);
		rank[8] = '\0';
	}

	return rank;
}

char* initGameSession(char* playerName)
{
	if (strncmp(playerName, GAME_PREFIX, strlen(GAME_PREFIX)) != 0) {
		// اگر قرار است همان نام را برگرداند باید مانند کیس بعدی, یک کپی عمیق از آن برگرداند تا در ادامه یک پوینتر دوبار آزاد نشود
		char* nameCopy = (char*)malloc((strlen(playerName) + 1) * sizeof(char));
		strcpy(nameCopy, playerName);
		return nameCopy;
	}

	int choice;
	printf(
		"Select game mode:\n"
		"0: Single Player\n"
		"1: Multiplayer\n"
		"2: Tournament\n"
		"3: Practice\n"
		"Enter choice: ");
	// بررسی مقدار بازگشتی scanf. اگر کاربر حرف وارد کند، choice مقداردهی نمی‌شود و برنامه رفتار نامشخص خواهد داشت
	if (scanf("%d", &choice) != 1) choice = 0;

	playerName = (char*)(malloc(sizeof(char) * 15));
	// در اینجا باید رشته ها را نال ترمینیت کنیم تا بتوانیم از توابع رشته ای استفاده کنیم و همچنین حافظه کافی برای رشته ها تخصیص دهیم
	switch (choice)
	{
	case(0):
		strncpy(playerName, "SINGLE", 6);
		playerName[6] = '\0';
		break;
	case(1):
		strncpy(playerName, "MULTI", 5);
		playerName[5] = '\0';
		break;
	case(2):
		strncpy(playerName, "TOURNAMENT", 10);
		playerName[10] = '\0';
		break;
	case(3):
		strncpy(playerName, "PRACTICE", 8);
		playerName[8] = '\0';
		break;
	default: // اضافه کردن default. اگر کاربر عددی غیر از 0-3 وارد کند، حافظه playerName مقداردهی نمی‌شد و چاپ آن باعث کرش می‌شد
		strncpy(playerName, "UNKNOWN", 8);
		playerName[7] = '\0';
		break;
	}
	return playerName;
}

void createLeaderboard(char** players, int* scores, int count)
{
	printf("\n=== LEADERBOARD ===\n");
	for (int i = 0; i < count; i++)
	{
		printf("Player %d: %s - Score: %d\n", i + 1, players[i], scores[i]);
	}
}

int main(int argc, char* argv[])
{
	srand(time(0)); // ایجاد یک بذر تصادفی برای تولید رشته های متفاوت در هر بار اجرای برنامه

	// اگر کاربر آرگومان وارد نکند، دسترسی به argv[1] باعث کرش می‌شود
	if (argc < 2) return 1;

	char* player = createPlayer(argv[1]);

	// اگر createPlayer نال برگرداند (مثلاً خطای حافظه)، نباید ادامه دهیم
	if (player == NULL) return 1;

	printf("Player created: %s\n", player);
	//free(player); این خط باید حذف شود تا در ادامه از پوینتر نامعابر استفاده نشود

	int* score = calculateScore(player);
	free(player); // حالا که دیگر نیازی به player نداریم می توانیم حافظه آن را آزاد کنیم

	// بررسی نال نبودن
	if (score != NULL) {
		printf("Player score: %d\n", *score);

		char* rank = getPlayerRank(score);
		printf("Player rank: %s\n", rank);

		char* session = initGameSession(rank);
		printf("Game session: %s\n", session);

		// انتقال آزادسازی به داخل بلوک برای جلوگیری از دابل فری یا خطا در صورت نال بودن
		free(score);
		free(rank);
		free(session);
	}

	return EXIT_SUCCESS;
}