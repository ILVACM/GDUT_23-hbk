#include<stdio.h>

int main()
{
    int n;
    scanf("%d", &n);

    switch (n)
    {
    case 1:
        printf("Good morning");
        break;

    case 2:
        printf("Good afternoon");
        break;

    case 3:
        printf("Good evening");
        break;

    case 4:
        printf("Good night");
        break;
    
    default:
        printf("Bye-bye");
        break;
    }

    return 0;
}