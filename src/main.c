#include <stddef.h>
#include "gui/gui.h"
#include "wordsearch/processor.h"


int main(int argc, char* argv[]){
    if(argc == 2) {
        // CLI mode: process image without GUI
        char* image_path = argv[1];
        return process_wordsearch_image(image_path, NULL);
    } else {
        // GUI mode
        return main_gui(argc, argv);
    }
}