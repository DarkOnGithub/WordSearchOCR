#include "gui/drawing_gui.h"
#include "nn/cnn.h"
#include "nn/core/tensor.h"
#include "nn/layers/cross_entropy_loss.h"
#include <gtk/gtk.h>
#include <glib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CANVAS_SIZE 392
#define GRID_SIZE 28
#define CELL_SIZE (CANVAS_SIZE / GRID_SIZE)

static guchar canvas_data[GRID_SIZE * GRID_SIZE];
static gboolean is_drawing = FALSE;
static gdouble last_x = -1, last_y = -1;
static int pen_size = 2;
static int noise_level = 0;

static GtkWidget *main_window;
static GtkWidget *drawing_area;
static GtkWidget *result_label;
static GtkWidget *confidence_label;
static GtkWidget *probabilities_box;
static GtkWidget *prob_labels[26];
static GtkWidget *prob_bars[26];
static GtkWidget *small_pen_button;
static GtkWidget *large_pen_button;
static GtkWidget *noise_scale;

static CNN *model = NULL;

static void clear_canvas(void);
static void predict_letter(void);
static gboolean on_draw(GtkWidget *widget, cairo_t *cr, gpointer data);
static gboolean on_button_press(GtkWidget *widget, GdkEventButton *event, gpointer data);
static gboolean on_button_release(GtkWidget *widget, GdkEventButton *event, gpointer data);
static gboolean on_motion_notify(GtkWidget *widget, GdkEventMotion *event, gpointer data);
static void on_clear_clicked(GtkWidget *widget, gpointer data);
static void on_predict_clicked(GtkWidget *widget, gpointer data);
static void on_small_pen_toggled(GtkWidget *widget, gpointer data);
static void on_large_pen_toggled(GtkWidget *widget, gpointer data);
static void on_noise_level_changed(GtkWidget *widget, gpointer data);
static void draw_point(int x, int y, int brush_size);
static void draw_line(int x0, int y0, int x1, int y1, int brush_size);
static void load_drawing_css(void);

static void load_drawing_css(void)
{
    GtkCssProvider *provider = gtk_css_provider_new();

    const char *css =
        "/* Drawing GUI Dark Theme */\n"
        ".drawing-window {\n"
        "  background-color: #0a0e1a;\n"
        "}\n"
        "\n"
        ".drawing-header {\n"
        "  background: linear-gradient(180deg, #141928 0%, #0f1422 100%);\n"
        "  border-bottom: 1px solid #1e2538;\n"
        "}\n"
        "\n"
        ".canvas-frame {\n"
        "  background-color: #ffffff;\n"
        "  border: 3px solid #3b5bdb;\n"
        "  border-radius: 12px;\n"
        "  padding: 4px;\n"
        "  box-shadow: 0 4px 20px rgba(59, 91, 219, 0.25),\n"
        "              inset 0 0 0 1px rgba(255,255,255,0.1);\n"
        "}\n"
        "\n"
        ".result-card {\n"
        "  background: linear-gradient(180deg, #141928 0%, #101420 100%);\n"
        "  border: 1px solid #242d48;\n"
        "  border-radius: 16px;\n"
        "  padding: 20px;\n"
        "}\n"
        "\n"
        ".result-letter {\n"
        "  font-size: 72px;\n"
        "  font-weight: bold;\n"
        "  color: #82aaff;\n"
        "  text-shadow: 0 0 20px rgba(130, 170, 255, 0.5);\n"
        "}\n"
        "\n"
        ".result-confidence {\n"
        "  font-size: 18px;\n"
        "  color: #8b98b8;\n"
        "}\n"
        "\n"
        ".prob-label {\n"
        "  font-family: 'JetBrains Mono', 'Fira Code', monospace;\n"
        "  font-size: 11px;\n"
        "  color: #7a8599;\n"
        "  min-width: 20px;\n"
        "}\n"
        "\n"
        ".prob-bar {\n"
        "  background-color: #1a2035;\n"
        "  border-radius: 3px;\n"
        "  min-height: 8px;\n"
        "}\n"
        "\n"
        ".prob-bar progress {\n"
        "  background: linear-gradient(90deg, #3b5bdb 0%, #5c7cfa 100%);\n"
        "  border-radius: 3px;\n"
        "}\n"
        "\n"
        ".clear-button {\n"
        "  background: linear-gradient(180deg, #c73e3e 0%, #a02e2e 100%);\n"
        "  border: 1px solid #8f2828;\n"
        "  color: #fff;\n"
        "  border-radius: 10px;\n"
        "  padding: 12px 28px;\n"
        "  font-weight: 600;\n"
        "  font-size: 14px;\n"
        "}\n"
        "\n"
        ".clear-button:hover {\n"
        "  background: linear-gradient(180deg, #d44848 0%, #b03535 100%);\n"
        "}\n"
        "\n"
        ".predict-button {\n"
        "  background: linear-gradient(180deg, #22a06b 0%, #16895a 100%);\n"
        "  border: 1px solid #0f7048;\n"
        "  color: #fff;\n"
        "  border-radius: 10px;\n"
        "  padding: 12px 28px;\n"
        "  font-weight: 600;\n"
        "  font-size: 14px;\n"
        "}\n"
        "\n"
        ".predict-button:hover {\n"
        "  background: linear-gradient(180deg, #28b878 0%, #1a9d65 100%);\n"
        "}\n"
        "\n"
        ".pen-size-button {\n"
        "  background: linear-gradient(180deg, #1a2035 0%, #242d48 100%);\n"
        "  border: 1px solid #3b5bdb;\n"
        "  color: #82aaff;\n"
        "  border-radius: 8px;\n"
        "  padding: 8px 16px;\n"
        "  font-weight: 600;\n"
        "  font-size: 12px;\n"
        "  min-width: 60px;\n"
        "  transition: all 0.2s ease;\n"
        "}\n"
        "\n"
        ".pen-size-button:hover {\n"
        "  background: linear-gradient(180deg, #242d48 0%, #2d3748 100%);\n"
        "  border-color: #5c7cfa;\n"
        "}\n"
        "\n"
        ".pen-size-button:checked {\n"
        "  background: linear-gradient(180deg, #3b5bdb 0%, #5c7cfa 100%);\n"
        "  border-color: #242d48;\n"
        "  color: #ffffff;\n"
        "  box-shadow: 0 2px 8px rgba(59, 91, 219, 0.4);\n"
        "}\n"
        "\n"
        ".pen-size-button:checked:hover {\n"
        "  background: linear-gradient(180deg, #5c7cfa 0%, #74a4ff 100%);\n"
        "}\n"
        "\n"
        ".noise-scale {\n"
        "  color: #82aaff;\n"
        "  font-size: 12px;\n"
        "  font-weight: 600;\n"
        "}\n"
        "\n"
        ".noise-scale trough {\n"
        "  background-color: #1a2035;\n"
        "  border: 1px solid #3b5bdb;\n"
        "  border-radius: 4px;\n"
        "  min-height: 8px;\n"
        "}\n"
        "\n"
        ".noise-scale slider {\n"
        "  background: linear-gradient(180deg, #3b5bdb 0%, #5c7cfa 100%);\n"
        "  border: 1px solid #242d48;\n"
        "  border-radius: 50%;\n"
        "  min-width: 16px;\n"
        "  min-height: 16px;\n"
        "  margin: -4px;\n"
        "}\n"
        "\n"
        ".noise-scale slider:hover {\n"
        "  background: linear-gradient(180deg, #5c7cfa 0%, #74a4ff 100%);\n"
        "}\n"
        "\n"
        ".noise-scale highlight {\n"
        "  background: linear-gradient(90deg, #3b5bdb 0%, #5c7cfa 100%);\n"
        "  border-radius: 4px;\n"
        "}\n"
        "\n"
        ".section-label {\n"
        "  color: #5a6580;\n"
        "  font-size: 12px;\n"
        "  font-weight: 600;\n"
        "  letter-spacing: 1px;\n"
        "}\n"
        "\n"
        ".instructions-label {\n"
        "  color: #6a7894;\n"
        "  font-size: 13px;\n"
        "}\n";

    gtk_css_provider_load_from_data(provider, css, -1, NULL);

    GdkScreen *screen = gdk_screen_get_default();
    gtk_style_context_add_provider_for_screen(
        screen, GTK_STYLE_PROVIDER(provider), GTK_STYLE_PROVIDER_PRIORITY_USER);
    g_object_unref(provider);

    GtkSettings *settings = gtk_settings_get_default();
    g_object_set(settings, "gtk-application-prefer-dark-theme", TRUE, NULL);
}


static void apply_noise_to_prediction_data(guchar *prediction_canvas)
{
    if (noise_level == 0) return;

    double normalized_level = (double)noise_level / 100.0;
    int num_noise_points = (int)((normalized_level * normalized_level) * GRID_SIZE * GRID_SIZE * 0.5);

    for (int i = 0; i < num_noise_points; i++) {
        int x = rand() % GRID_SIZE;
        int y = rand() % GRID_SIZE;
        int index = y * GRID_SIZE + x;
        guchar gray_level = (guchar)((double)rand() / RAND_MAX * 180);
        prediction_canvas[index] = gray_level;
    }
}

static void clear_canvas(void)
{
    memset(canvas_data, 0, sizeof(canvas_data));
    noise_level = 0;

    if (result_label && GTK_IS_LABEL(result_label)) {
        gtk_label_set_text(GTK_LABEL(result_label), "?");
    }
    if (confidence_label && GTK_IS_LABEL(confidence_label)) {
        gtk_label_set_text(GTK_LABEL(confidence_label), "Draw a letter and click Predict");
    }

    for (int i = 0; i < 26; i++) {
        if (prob_bars[i] && GTK_IS_PROGRESS_BAR(prob_bars[i])) {
            gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(prob_bars[i]), 0.0);
        }
    }

    if (noise_scale && GTK_IS_RANGE(noise_scale)) {
        gtk_range_set_value(GTK_RANGE(noise_scale), 0.0);
    }

    if (drawing_area && GTK_IS_WIDGET(drawing_area)) {
        gtk_widget_queue_draw(drawing_area);
    }
}

static void draw_point(int grid_x, int grid_y, int brush_size)
{
    if (brush_size == 1) {
        if (grid_x >= 0 && grid_x < GRID_SIZE && grid_y >= 0 && grid_y < GRID_SIZE) {
            canvas_data[grid_y * GRID_SIZE + grid_x] = 255;
        }
        return;
    }

    int actual_brush_size = (brush_size == 2) ? 1 : brush_size;

    for (int dy = -actual_brush_size; dy <= actual_brush_size; dy++) {
        for (int dx = -actual_brush_size; dx <= actual_brush_size; dx++) {
            int nx = grid_x + dx;
            int ny = grid_y + dy;

            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE) {
                float dist = sqrtf((float)(dx * dx + dy * dy));
                if (dist <= actual_brush_size + 0.5f) {
                    float intensity = 1.0f - (dist / (actual_brush_size + 1.0f)) * 0.3f;
                    int current = canvas_data[ny * GRID_SIZE + nx];
                    int new_val = (int)(255 * intensity);
                    canvas_data[ny * GRID_SIZE + nx] = (guchar)(current > new_val ? current : new_val);
                }
            }
        }
    }
}

static void draw_line(int x0, int y0, int x1, int y1, int brush_size)
{
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        draw_point(x0, y0, brush_size);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static gboolean on_draw(GtkWidget *widget, cairo_t *cr, gpointer data)
{
    (void)widget;
    (void)data;

    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            guchar val = canvas_data[y * GRID_SIZE + x];
            double gray = 1.0 - (val / 255.0);
            cairo_set_source_rgb(cr, gray, gray, gray);
            cairo_rectangle(cr, x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            cairo_fill(cr);
        }
    }

    if (noise_level > 0) {
        double normalized_level = (double)noise_level / 100.0;
        int num_noise_points = (int)((normalized_level * normalized_level) * GRID_SIZE * GRID_SIZE * 0.5);

        for (int i = 0; i < num_noise_points; i++) {
            int x = rand() % GRID_SIZE;
            int y = rand() % GRID_SIZE;
            double gray_level = (double)rand() / RAND_MAX * 0.7;
            cairo_set_source_rgb(cr, gray_level, gray_level, gray_level);
            cairo_rectangle(cr, x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            cairo_fill(cr);
        }
    }

    cairo_set_source_rgba(cr, 0.85, 0.85, 0.85, 0.3);
    cairo_set_line_width(cr, 0.5);

    for (int i = 0; i <= GRID_SIZE; i++) {
        cairo_move_to(cr, i * CELL_SIZE, 0);
        cairo_line_to(cr, i * CELL_SIZE, CANVAS_SIZE);
        cairo_move_to(cr, 0, i * CELL_SIZE);
        cairo_line_to(cr, CANVAS_SIZE, i * CELL_SIZE);
    }
    cairo_stroke(cr);

    return FALSE;
}

static gboolean on_button_press(GtkWidget *widget, GdkEventButton *event, gpointer data)
{
    (void)widget;
    (void)data;

    if (event->button == 1) {
        is_drawing = TRUE;

        int grid_x = (int)(event->x / CELL_SIZE);
        int grid_y = (int)(event->y / CELL_SIZE);

        if (grid_x >= 0 && grid_x < GRID_SIZE && grid_y >= 0 && grid_y < GRID_SIZE) {
            draw_point(grid_x, grid_y, pen_size);
            last_x = event->x;
            last_y = event->y;
            gtk_widget_queue_draw(drawing_area);
        }
    }

    return TRUE;
}

static gboolean on_button_release(GtkWidget *widget, GdkEventButton *event, gpointer data)
{
    (void)widget;
    (void)event;
    (void)data;

    is_drawing = FALSE;
    last_x = -1;
    last_y = -1;

    return TRUE;
}

static gboolean on_motion_notify(GtkWidget *widget, GdkEventMotion *event, gpointer data)
{
    (void)widget;
    (void)data;

    if (is_drawing) {
        int grid_x = (int)(event->x / CELL_SIZE);
        int grid_y = (int)(event->y / CELL_SIZE);

        if (last_x >= 0 && last_y >= 0) {
            int last_grid_x = (int)(last_x / CELL_SIZE);
            int last_grid_y = (int)(last_y / CELL_SIZE);

            draw_line(last_grid_x, last_grid_y, grid_x, grid_y, pen_size);
        } else {
            draw_point(grid_x, grid_y, pen_size);
        }

        last_x = event->x;
        last_y = event->y;
        gtk_widget_queue_draw(drawing_area);
    }

    return TRUE;
}

static void on_clear_clicked(GtkWidget *widget, gpointer data)
{
    (void)widget;
    (void)data;
    clear_canvas();
}

static void on_small_pen_toggled(GtkWidget *widget, gpointer data)
{
    (void)widget;
    (void)data;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))) {
        pen_size = 1;
        // Turn off the large button
        g_signal_handlers_block_by_func(large_pen_button, on_large_pen_toggled, NULL);
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(large_pen_button), FALSE);
        g_signal_handlers_unblock_by_func(large_pen_button, on_large_pen_toggled, NULL);
    }
}

static void on_large_pen_toggled(GtkWidget *widget, gpointer data)
{
    (void)widget;
    (void)data;
    if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget))) {
        pen_size = 2;
        // Turn off the small button
        g_signal_handlers_block_by_func(small_pen_button, on_small_pen_toggled, NULL);
        gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(small_pen_button), FALSE);
        g_signal_handlers_unblock_by_func(small_pen_button, on_small_pen_toggled, NULL);
    }
}

static void on_noise_level_changed(GtkWidget *widget, gpointer data)
{
    (void)widget;
    (void)data;
    noise_level = (int)gtk_range_get_value(GTK_RANGE(widget));
    if (drawing_area && GTK_IS_WIDGET(drawing_area)) {
        gtk_widget_queue_draw(drawing_area);
    }
}

static void predict_letter(void)
{
    if (!model) {
        gtk_label_set_text(GTK_LABEL(result_label), "!");
        gtk_label_set_text(GTK_LABEL(confidence_label), "Model not loaded");
        return;
    }

    int input_shape[] = {1, 1, 28, 28};
    Tensor *input = tensor_create(input_shape, 4);

    if (!input) {
        gtk_label_set_text(GTK_LABEL(result_label), "!");
        gtk_label_set_text(GTK_LABEL(confidence_label), "Failed to create tensor");
        return;
    }

    guchar prediction_canvas[GRID_SIZE * GRID_SIZE];
    memcpy(prediction_canvas, canvas_data, sizeof(canvas_data));
    apply_noise_to_prediction_data(prediction_canvas);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            float pixel_value = 1.0f - ((float)prediction_canvas[y * 28 + x] / 255.0f);
            pixel_value = (pixel_value - 0.5f) / 0.5f;
            input->data[y * 28 + x] = pixel_value;
        }
    }

    CNNForwardResult *forward_result = cnn_forward(model, input);

    if (!forward_result) {
        gtk_label_set_text(GTK_LABEL(result_label), "!");
        gtk_label_set_text(GTK_LABEL(confidence_label), "Forward pass failed");
        tensor_free(input);
        return;
    }

    Tensor *probs = softmax(forward_result->fc2_out);

    if (!probs) {
        gtk_label_set_text(GTK_LABEL(result_label), "!");
        gtk_label_set_text(GTK_LABEL(confidence_label), "Softmax failed");
        cnn_forward_result_free(forward_result);
        tensor_free(input);
        return;
    }

    int max_idx = 0;
    float max_prob = probs->data[0];

    for (int i = 1; i < 26; i++) {
        if (probs->data[i] > max_prob) {
            max_prob = probs->data[i];
            max_idx = i;
        }
    }

    char letter[2] = {(char)('A' + max_idx), '\0'};
    gtk_label_set_text(GTK_LABEL(result_label), letter);

    char confidence_text[64];
    snprintf(confidence_text, sizeof(confidence_text), "Confidence: %.1f%%", max_prob * 100.0f);
    gtk_label_set_text(GTK_LABEL(confidence_label), confidence_text);

    for (int i = 0; i < 26; i++) {
        if (prob_bars[i] && GTK_IS_PROGRESS_BAR(prob_bars[i])) {
            gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(prob_bars[i]), probs->data[i]);
        }
    }

    tensor_free(probs);
    cnn_forward_result_free(forward_result);
    tensor_free(input);
}

static void on_predict_clicked(GtkWidget *widget, gpointer data)
{
    (void)widget;
    (void)data;
    predict_letter();
}

int main_drawing_gui(int argc, char *argv[])
{
    gtk_init(&argc, &argv);

    load_drawing_css();

    printf("Loading CNN model...\n");
    model = cnn_create();
    if (cnn_load_weights(model, 12) != 0) {
        fprintf(stderr, "Warning: Failed to load CNN weights\n");
    }
    cnn_eval(model);
    printf("CNN model loaded successfully.\n");

    srand(time(NULL));
    clear_canvas();

    main_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(main_window), "CNN Letter Classifier");
    gtk_window_set_default_size(GTK_WINDOW(main_window), 800, 560);
    gtk_window_set_resizable(GTK_WINDOW(main_window), FALSE);
    g_signal_connect(main_window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_style_context_add_class(gtk_widget_get_style_context(main_window), "drawing-window");


    GtkWidget *main_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 30);
    gtk_widget_set_margin_start(main_hbox, 30);
    gtk_widget_set_margin_end(main_hbox, 30);
    gtk_widget_set_margin_top(main_hbox, 25);
    gtk_widget_set_margin_bottom(main_hbox, 25);
    gtk_container_add(GTK_CONTAINER(main_window), main_hbox);

    GtkWidget *left_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    gtk_box_pack_start(GTK_BOX(main_hbox), left_vbox, FALSE, FALSE, 0);

    GtkWidget *canvas_label = gtk_label_new("DRAW HERE");
    gtk_style_context_add_class(gtk_widget_get_style_context(canvas_label), "section-label");
    gtk_box_pack_start(GTK_BOX(left_vbox), canvas_label, FALSE, FALSE, 0);

    GtkWidget *canvas_frame = gtk_frame_new(NULL);
    gtk_frame_set_shadow_type(GTK_FRAME(canvas_frame), GTK_SHADOW_NONE);
    gtk_style_context_add_class(gtk_widget_get_style_context(canvas_frame), "canvas-frame");
    gtk_box_pack_start(GTK_BOX(left_vbox), canvas_frame, FALSE, FALSE, 0);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, CANVAS_SIZE, CANVAS_SIZE);
    gtk_widget_add_events(drawing_area,
                          GDK_BUTTON_PRESS_MASK |
                          GDK_BUTTON_RELEASE_MASK |
                          GDK_POINTER_MOTION_MASK);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(on_draw), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(on_button_press), NULL);
    g_signal_connect(drawing_area, "button-release-event", G_CALLBACK(on_button_release), NULL);
    g_signal_connect(drawing_area, "motion-notify-event", G_CALLBACK(on_motion_notify), NULL);
    gtk_container_add(GTK_CONTAINER(canvas_frame), drawing_area);

    GtkWidget *pen_size_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_halign(pen_size_box, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(left_vbox), pen_size_box, FALSE, FALSE, 10);

    GtkWidget *pen_size_label = gtk_label_new("Pen Size:");
    gtk_style_context_add_class(gtk_widget_get_style_context(pen_size_label), "section-label");
    gtk_box_pack_start(GTK_BOX(pen_size_box), pen_size_label, FALSE, FALSE, 0);

    small_pen_button = gtk_toggle_button_new_with_label("Small");
    gtk_style_context_add_class(gtk_widget_get_style_context(small_pen_button), "pen-size-button");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(small_pen_button), pen_size == 1);
    g_signal_connect(small_pen_button, "toggled", G_CALLBACK(on_small_pen_toggled), NULL);
    gtk_box_pack_start(GTK_BOX(pen_size_box), small_pen_button, FALSE, FALSE, 0);

    large_pen_button = gtk_toggle_button_new_with_label("Large");
    gtk_style_context_add_class(gtk_widget_get_style_context(large_pen_button), "pen-size-button");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(large_pen_button), pen_size == 2);
    g_signal_connect(large_pen_button, "toggled", G_CALLBACK(on_large_pen_toggled), NULL);
    gtk_box_pack_start(GTK_BOX(pen_size_box), large_pen_button, FALSE, FALSE, 0);

    GtkWidget *noise_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_widget_set_halign(noise_box, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(left_vbox), noise_box, FALSE, FALSE, 10);

    GtkWidget *noise_label = gtk_label_new("Noise Level:");
    gtk_style_context_add_class(gtk_widget_get_style_context(noise_label), "section-label");
    gtk_box_pack_start(GTK_BOX(noise_box), noise_label, FALSE, FALSE, 0);

    noise_scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, 100, 1);
    gtk_scale_set_value_pos(GTK_SCALE(noise_scale), GTK_POS_RIGHT);
    gtk_range_set_value(GTK_RANGE(noise_scale), noise_level);
    gtk_widget_set_size_request(noise_scale, 120, -1);
    gtk_style_context_add_class(gtk_widget_get_style_context(noise_scale), "noise-scale");
    g_signal_connect(noise_scale, "value-changed", G_CALLBACK(on_noise_level_changed), NULL);
    gtk_box_pack_start(GTK_BOX(noise_box), noise_scale, FALSE, FALSE, 0);

    GtkWidget *button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 15);
    gtk_widget_set_halign(button_box, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(left_vbox), button_box, FALSE, FALSE, 5);

    GtkWidget *clear_button = gtk_button_new_with_label("Clear");
    gtk_style_context_add_class(gtk_widget_get_style_context(clear_button), "clear-button");
    g_signal_connect(clear_button, "clicked", G_CALLBACK(on_clear_clicked), NULL);
    gtk_box_pack_start(GTK_BOX(button_box), clear_button, FALSE, FALSE, 0);

    GtkWidget *predict_button = gtk_button_new_with_label("Predict");
    gtk_style_context_add_class(gtk_widget_get_style_context(predict_button), "predict-button");
    g_signal_connect(predict_button, "clicked", G_CALLBACK(on_predict_clicked), NULL);
    gtk_box_pack_start(GTK_BOX(button_box), predict_button, FALSE, FALSE, 0);

    GtkWidget *right_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    gtk_box_pack_start(GTK_BOX(main_hbox), right_vbox, TRUE, TRUE, 0);

    GtkWidget *result_section_label = gtk_label_new("PREDICTION");
    gtk_style_context_add_class(gtk_widget_get_style_context(result_section_label), "section-label");
    gtk_box_pack_start(GTK_BOX(right_vbox), result_section_label, FALSE, FALSE, 0);

    GtkWidget *result_card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    gtk_style_context_add_class(gtk_widget_get_style_context(result_card), "result-card");
    gtk_widget_set_halign(result_card, GTK_ALIGN_FILL);
    gtk_box_pack_start(GTK_BOX(right_vbox), result_card, FALSE, FALSE, 0);

    result_label = gtk_label_new("?");
    gtk_style_context_add_class(gtk_widget_get_style_context(result_label), "result-letter");
    gtk_widget_set_halign(result_label, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(result_card), result_label, FALSE, FALSE, 10);

    confidence_label = gtk_label_new("Draw a letter and click Predict");
    gtk_style_context_add_class(gtk_widget_get_style_context(confidence_label), "result-confidence");
    gtk_widget_set_halign(confidence_label, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(result_card), confidence_label, FALSE, FALSE, 5);

    GtkWidget *prob_label = gtk_label_new("ALL PROBABILITIES");
    gtk_style_context_add_class(gtk_widget_get_style_context(prob_label), "section-label");
    gtk_widget_set_margin_top(prob_label, 10);
    gtk_box_pack_start(GTK_BOX(right_vbox), prob_label, FALSE, FALSE, 0);

    GtkWidget *prob_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(prob_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_vexpand(prob_scroll, TRUE);
    gtk_box_pack_start(GTK_BOX(right_vbox), prob_scroll, TRUE, TRUE, 0);

    probabilities_box = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(probabilities_box), 4);
    gtk_grid_set_column_spacing(GTK_GRID(probabilities_box), 8);
    gtk_widget_set_margin_start(probabilities_box, 5);
    gtk_widget_set_margin_end(probabilities_box, 5);
    gtk_container_add(GTK_CONTAINER(prob_scroll), probabilities_box);

    for (int i = 0; i < 26; i++) {
        int row = i % 13;
        int col_offset = (i / 13) * 3;

        char label_text[4];
        snprintf(label_text, sizeof(label_text), "%c:", 'A' + i);

        prob_labels[i] = gtk_label_new(label_text);
        gtk_style_context_add_class(gtk_widget_get_style_context(prob_labels[i]), "prob-label");
        gtk_widget_set_halign(prob_labels[i], GTK_ALIGN_END);
        gtk_grid_attach(GTK_GRID(probabilities_box), prob_labels[i], col_offset, row, 1, 1);

        prob_bars[i] = gtk_progress_bar_new();
        gtk_style_context_add_class(gtk_widget_get_style_context(prob_bars[i]), "prob-bar");
        gtk_widget_set_size_request(prob_bars[i], 100, 12);
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(prob_bars[i]), 0.0);
        gtk_grid_attach(GTK_GRID(probabilities_box), prob_bars[i], col_offset + 1, row, 1, 1);
    }



    gtk_widget_show_all(main_window);

    gtk_main();

    if (model) {
        cnn_free(model);
        model = NULL;
    }

    return 0;
}

