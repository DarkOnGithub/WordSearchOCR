#include <stdio.h>
#include <stdlib.h>
#include "ocr/wordsearch_processor.h"


int main(int argc, char* argv[]){
    if(argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        exit(1);
    }
    char* image_path = argv[1];
    process_wordsearch_image(image_path);
    return 0;
}