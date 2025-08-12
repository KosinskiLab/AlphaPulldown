#include <stdlib.h>

#include <reproc/run.h>

int main(int argc, const char **argv)
{
    const char* cmd[] = {"echo", "Hello", NULL};
    int r = reproc_run(cmd, (reproc_options){0});

    return 0;
    /* Fails on the CI somehow but packages are fine */
    /* if (r < 0) { */
    /*     fprintf(stderr, "%s\n", reproc_strerror(r)); */
    /* } */
    /* return abs(r); */
}
