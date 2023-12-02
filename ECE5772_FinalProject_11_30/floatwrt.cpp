
#include <stdio.h>
#include <math.h>

int main() {
    
    FILE *fptr = fopen("sample.txt", "w");
    
    if (fptr == NULL){
        printf("Couldn't open file");
        return 0;
    }
    
    for (int i = 0; i < 100; i++){
        fprintf(fptr, "%0.5lf\n", log((double)i+1)); 
    }
    
    fclose(fptr);
    return 0;
    
}
