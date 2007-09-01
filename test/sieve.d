/* Eratosthenes Sieve prime number calculation. */

bool flags[8191];

int main()
{   int     i, prime, k, count, iter;

    printf("10 iterations\n");
    for (iter = 1;
    iter <= 10;
    iter++)
    {
    count = 0;
    flags[] = true;
    for (i = 0; i < flags.length; i++)
    {   if (flags[i])
        {
        prime = i + i + 3;
        k = i + prime;
        while (k < flags.length)
        {
            flags[k] = false;
            k += prime;
        }
        count += 1;
        }
    }
    }
    printf("%d primes\n", count);
    return 0;
}
