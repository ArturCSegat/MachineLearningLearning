#include <stdio.h>
#include <stdlib.h>
#include <time.h>

# define TO_RADIANS 3.14159/180.0
# define BATCH 100

float f(float x) {
//    if (sin(x) == 0) {
//        return 0;
//    }
//    return cos(x) / sin(x);

    return 1.5 * x + 0.69 ;//+ (float)rand()/(float)(RAND_MAX);
}

struct example {
    float x;
    float label;
};

void print_example(struct example e) {
    printf("example{x: %f, label: %f}\n", e.x, e.label);
}

typedef struct {
    float a;
    float b;
} Regressor;

float linear(Regressor r, float x) {
    return x * r.a + r.b;
}

Regressor new_regressor() {
    float a =(float)rand()/(float)(RAND_MAX/3);
    float b =(float)rand()/(float)(RAND_MAX)/3;
    return (Regressor){a, b};
}

void run_examples(Regressor* r, struct example e[BATCH]) {
    float avg_x = 0;
    float avg_label = 0;
    
    for (int i = 0; i < BATCH; i++) {
        avg_x += e[i].x;
        avg_label += e[i].label;
    }
    avg_x /= BATCH;
    avg_label /= BATCH;

    float top = 0;
    float bot = 0;
    for (int i = 0; i < BATCH; i++) {
        top += (e[i].x - avg_x) * (e[i].label - avg_label);
        bot += (e[i].x - avg_x) * (e[i].x - avg_x);
    }
    float a_coef = top / bot;
    float b_coef = avg_label - (a_coef * avg_x);
    
    r->a = a_coef;
    r->b = b_coef;
}

void print_regressor(Regressor r) {
    printf("Regressor{a: %f, b: %f}\n", r.a, r.b);
}

int main() {
    srand(time(NULL));
    
    struct example examples[100];
    
    for (int i = 0; i < 100; i++) {
        examples[i].x = i;
        examples[i].label = f((float)i * TO_RADIANS);
        // print_example(examples[i]);
    }
    
    Regressor r = new_regressor();
    print_regressor(r);
    run_examples(&r, examples);
    print_regressor(r);

    for (int i = 0; i < 10; i++) {
        printf("Example: %d\n", i + 1);
        print_example(examples[i]);
        print_example((struct example){examples[i].x, linear(r, examples[i].x)});
    }

    return 0;
}
