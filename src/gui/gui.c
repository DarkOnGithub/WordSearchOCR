#include "wordsearch/processor.h"
#include "wordsearch/word_detection.h"
#include "nn/cnn.h"
#include "solver/solver.h"
#include "nn/inference.h"
#include "image/operations.h"
#include <glib.h>
#include <gtk/gtk.h>

static GtkWidget *main_window;
static GtkWidget *image_display;
static GtkWidget *right_image_display;
static GtkWidget *load_button;
static GtkWidget *process_button;
static GtkWidget *paned;
static GtkWidget *image_container;
static GtkWidget *results_container;
static GtkWidget *notebook;
static gchar *current_image_path = NULL;
static gboolean is_processing = FALSE;
typedef struct
{
    char *step_name;
    char *filename;
} ButtonInfo;

static void free_button_info(gpointer data)
{
    ButtonInfo *info = (ButtonInfo *)data;
    if (info)
    {
        g_free(info->step_name);
        g_free(info->filename);
        g_free(info);
    }
}

static int current_processing_mode = 0;
static GtkWidget *tab_containers[3] = {NULL, NULL, NULL};
static gboolean tab_processed[3] = {FALSE, FALSE, FALSE};
static GList *tab_buttons[3] = {NULL, NULL, NULL};
static GtkWidget *selected_processing_button = NULL;

static void load_image_callback(GtkWidget *widget, gpointer data);
static void process_callback(GtkWidget *widget, gpointer data);
static void create_processing_step_button(const char *step_name,
                                          const char *filename);
static void create_preview_button(const char *button_text, const char *filename);
static void processing_step_callback(GtkWidget *widget, gpointer data);
static void notebook_switch_page_callback(GtkNotebook *notebook,
                                          GtkWidget *page, guint page_num,
                                          gpointer user_data);
static void setup_initial_right_panel(const gchar *image_path);
static void show_file_chooser(void);
static void display_image(const gchar *image_path);
static void display_right_image(const gchar *image_path);
static void process_both_modes(const gchar *image_path);
static void animate_image_transition(void);
static void load_dark_theme_css(void);
static void show_welcome_panel(void);

int main_gui(int argc, char *argv[])
{
    GtkWidget *header_bar;
    GtkWidget *results_label;

    gtk_init(&argc, &argv);

    load_dark_theme_css();

    main_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(main_window), "WordSearch OCR");
    gtk_window_set_default_size(GTK_WINDOW(main_window), 1400, 900);
    g_signal_connect(main_window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    GtkStyleContext *win_ctx = gtk_widget_get_style_context(main_window);
    gtk_style_context_add_class(win_ctx, "app-window");

    header_bar = gtk_header_bar_new();
    gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(header_bar), TRUE);
    gtk_header_bar_set_title(GTK_HEADER_BAR(header_bar), "WordSearch OCR");

    load_button = gtk_button_new_with_label("Load Image");
    g_signal_connect(load_button, "clicked", G_CALLBACK(load_image_callback),
                     NULL);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(header_bar), load_button);

    process_button = gtk_button_new_with_label("Process Image");
    gtk_widget_set_sensitive(process_button, FALSE);
    g_signal_connect(process_button, "clicked", G_CALLBACK(process_callback),
                     NULL);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(header_bar), process_button);

    gtk_style_context_add_class(gtk_widget_get_style_context(header_bar),
                                "app-header");
    gtk_style_context_add_class(gtk_widget_get_style_context(load_button),
                                "primary-button");
    gtk_style_context_add_class(gtk_widget_get_style_context(process_button),
                                "accent-button");

    gtk_window_set_titlebar(GTK_WINDOW(main_window), header_bar);

    paned = gtk_paned_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_container_add(GTK_CONTAINER(main_window), paned);
    gtk_style_context_add_class(gtk_widget_get_style_context(paned),
                                "main-paned");

    image_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    gtk_widget_set_margin_start(image_container, 10);
    gtk_widget_set_margin_end(image_container, 10);
    gtk_widget_set_margin_top(image_container, 10);
    gtk_widget_set_margin_bottom(image_container, 10);
    gtk_style_context_add_class(gtk_widget_get_style_context(image_container),
                                "panel-left");
    gtk_paned_pack1(GTK_PANED(paned), image_container, TRUE, FALSE);

    GtkWidget *original_section = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_box_pack_start(GTK_BOX(image_container), original_section, FALSE, FALSE,
                       0);

    GtkWidget *original_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(original_title),
                         "<b><span size='large'>Original Image</span></b>");
    gtk_style_context_add_class(gtk_widget_get_style_context(original_title),
                                "section-title");
    gtk_box_pack_start(GTK_BOX(original_section), original_title, FALSE, FALSE,
                       5);

    GtkWidget *image_frame = gtk_frame_new(NULL);
    gtk_frame_set_shadow_type(GTK_FRAME(image_frame), GTK_SHADOW_ETCHED_IN);
    gtk_widget_set_margin_bottom(image_frame, 10);
    gtk_style_context_add_class(gtk_widget_get_style_context(image_frame),
                                "card-frame");
    gtk_box_pack_start(GTK_BOX(original_section), image_frame, FALSE, FALSE, 0);

    image_display = gtk_image_new();
    gtk_widget_set_size_request(image_display, 250, 200);
    gtk_container_add(GTK_CONTAINER(image_frame), image_display);

    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_widget_set_margin_top(separator, 10);
    gtk_widget_set_margin_bottom(separator, 10);
    gtk_style_context_add_class(gtk_widget_get_style_context(separator),
                                "thin-separator");
    gtk_box_pack_start(GTK_BOX(image_container), separator, FALSE, FALSE, 0);

    notebook = gtk_notebook_new();
    gtk_box_pack_start(GTK_BOX(image_container), notebook, TRUE, TRUE, 0);
    gtk_style_context_add_class(gtk_widget_get_style_context(notebook),
                                "processing-notebook");

    GtkWidget *grid_processing_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(grid_processing_box), 10);

    GtkWidget *grid_title = gtk_label_new(NULL);
    gtk_label_set_markup(
        GTK_LABEL(grid_title),
        "<b><span size='large'>Grid Processing Steps</span></b>");
    gtk_style_context_add_class(gtk_widget_get_style_context(grid_title),
                                "section-title");
    gtk_box_pack_start(GTK_BOX(grid_processing_box), grid_title, FALSE, FALSE,
                       5);

    GtkWidget *grid_tab_label = gtk_label_new("Grid Processing");
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), grid_processing_box,
                             grid_tab_label);

    GtkWidget *word_detection_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(word_detection_box), 10);

    GtkWidget *word_title = gtk_label_new(NULL);
    gtk_label_set_markup(
        GTK_LABEL(word_title),
        "<b><span size='large'>Word Detection Steps</span></b>");
    gtk_style_context_add_class(gtk_widget_get_style_context(word_title),
                                "section-title");
    gtk_box_pack_start(GTK_BOX(word_detection_box), word_title, FALSE, FALSE,
                       5);

    GtkWidget *word_tab_label = gtk_label_new("Word Detection");
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), word_detection_box,
                             word_tab_label);

    GtkWidget *preview_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(preview_box), 10);

    GtkWidget *preview_title = gtk_label_new(NULL);
    gtk_label_set_markup(
        GTK_LABEL(preview_title),
        "<b><span size='large'>Preview Results</span></b>");
    gtk_style_context_add_class(gtk_widget_get_style_context(preview_title),
                                "section-title");
    gtk_box_pack_start(GTK_BOX(preview_box), preview_title, FALSE, FALSE,
                       5);

    GtkWidget *preview_tab_label = gtk_label_new("Preview");
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), preview_box,
                             preview_tab_label);

    g_signal_connect(notebook, "switch-page",
                     G_CALLBACK(notebook_switch_page_callback), NULL);

    results_container = gtk_box_new(GTK_ORIENTATION_VERTICAL, 15);
    gtk_widget_set_margin_start(results_container, 10);
    gtk_widget_set_margin_end(results_container, 10);
    gtk_widget_set_margin_top(results_container, 10);
    gtk_widget_set_margin_bottom(results_container, 10);
    gtk_style_context_add_class(gtk_widget_get_style_context(results_container),
                                "panel-right");
    gtk_paned_pack2(GTK_PANED(paned), results_container, TRUE, FALSE);

    show_welcome_panel();

    gtk_paned_set_position(GTK_PANED(paned), 380);

    gtk_widget_show_all(main_window);

    gtk_main();

    if (current_image_path)
    {
        g_free(current_image_path);
    }

    return 0;
}

static void notebook_switch_page_callback(GtkNotebook *notebook,
                                          GtkWidget *page, guint page_num,
                                          gpointer user_data)
{
    if (selected_processing_button != NULL &&
        GTK_IS_WIDGET(selected_processing_button))
    {
        GtkStyleContext *prev_ctx =
            gtk_widget_get_style_context(selected_processing_button);
        gtk_style_context_remove_class(prev_ctx, "selected");
    }

    current_processing_mode = page_num;
    selected_processing_button = NULL;

    if (current_image_path)
    {
        setup_initial_right_panel(current_image_path);
    }
}

static void load_image_callback(GtkWidget *widget, gpointer data)
{
    show_file_chooser();
}

static void show_file_chooser(void)
{
    GtkWidget *dialog;
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    gint res;

    dialog = gtk_file_chooser_dialog_new("Open Image", GTK_WINDOW(main_window),
                                         action, "_Cancel", GTK_RESPONSE_CANCEL,
                                         "_Open", GTK_RESPONSE_ACCEPT, NULL);

    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(dialog), "images");

    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Image files");
    gtk_file_filter_add_mime_type(filter, "image/jpeg");
    gtk_file_filter_add_mime_type(filter, "image/png");
    gtk_file_filter_add_mime_type(filter, "image/bmp");
    gtk_file_filter_add_mime_type(filter, "image/tiff");
    gtk_file_filter_add_pattern(filter, "*.jpg");
    gtk_file_filter_add_pattern(filter, "*.jpeg");
    gtk_file_filter_add_pattern(filter, "*.png");
    gtk_file_filter_add_pattern(filter, "*.bmp");
    gtk_file_filter_add_pattern(filter, "*.tiff");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    res = gtk_dialog_run(GTK_DIALOG(dialog));
    if (res == GTK_RESPONSE_ACCEPT)
    {
        GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
        gchar *filename = gtk_file_chooser_get_filename(chooser);

        if (current_image_path)
        {
            g_free(current_image_path);
        }
        current_image_path = filename;

        display_image(current_image_path);

        gtk_widget_set_sensitive(process_button, TRUE);

        is_processing = FALSE;

        if (selected_processing_button != NULL &&
            GTK_IS_WIDGET(selected_processing_button))
        {
            GtkStyleContext *ctx =
                gtk_widget_get_style_context(selected_processing_button);
            gtk_style_context_remove_class(ctx, "selected");
        }
        selected_processing_button = NULL;

        for (int i = 0; i < 3; i++)
        {
            tab_processed[i] = FALSE;
            if (tab_containers[i] != NULL)
            {
                gtk_widget_destroy(tab_containers[i]);
                tab_containers[i] = NULL;
            }
            if (tab_buttons[i] != NULL)
            {
                g_list_free_full(tab_buttons[i], free_button_info);
                tab_buttons[i] = NULL;
            }
        }
    }

    gtk_widget_destroy(dialog);
}

static void display_image(const gchar *image_path)
{
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(image_path, NULL);
    if (pixbuf)
    {
        int container_width = 250;
        int container_height = 200;
        int width = gdk_pixbuf_get_width(pixbuf);
        int height = gdk_pixbuf_get_height(pixbuf);

        double scale_x = (double)container_width / width;
        double scale_y = (double)container_height / height;
        double scale = MIN(scale_x, scale_y);

        int new_width = (int)(width * scale);
        int new_height = (int)(height * scale);

        GdkPixbuf *scaled_pixbuf = gdk_pixbuf_scale_simple(
            pixbuf, new_width, new_height, GDK_INTERP_BILINEAR);
        gtk_image_set_from_pixbuf(GTK_IMAGE(image_display), scaled_pixbuf);
        g_object_unref(scaled_pixbuf);

        setup_initial_right_panel(image_path);

        g_object_unref(pixbuf);
    }
}

static void setup_initial_right_panel(const gchar *image_path)
{
    GList *children =
        gtk_container_get_children(GTK_CONTAINER(results_container));
    for (GList *iter = children; iter != NULL; iter = iter->next)
    {
        gtk_widget_destroy(GTK_WIDGET(iter->data));
    }
    g_list_free(children);

    GtkWidget *section_title = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(section_title),
                         "<b><span size='x-large'>Image Preview</span></b>");
    gtk_style_context_add_class(gtk_widget_get_style_context(section_title),
                                "section-title");
    gtk_box_pack_start(GTK_BOX(results_container), section_title, FALSE, FALSE,
                       10);

    GtkWidget *right_image_frame = gtk_frame_new(NULL);
    gtk_frame_set_shadow_type(GTK_FRAME(right_image_frame),
                              GTK_SHADOW_ETCHED_IN);
    gtk_widget_set_margin_bottom(right_image_frame, 15);
    gtk_style_context_add_class(gtk_widget_get_style_context(right_image_frame),
                                "card-frame");
    gtk_box_pack_start(GTK_BOX(results_container), right_image_frame, FALSE,
                       FALSE, 0);

    right_image_display = gtk_image_new();
    gtk_widget_set_size_request(right_image_display, 800, 600);
    gtk_container_add(GTK_CONTAINER(right_image_frame), right_image_display);

    GtkWidget *image_label = gtk_label_new(NULL);
    gtk_label_set_line_wrap(GTK_LABEL(image_label), TRUE);
    gtk_label_set_markup(GTK_LABEL(image_label),
                         "<i><span size='large'>Original Image</span></i>\n\n");
    gtk_label_set_justify(GTK_LABEL(image_label), GTK_JUSTIFY_CENTER);
    gtk_box_pack_start(GTK_BOX(results_container), image_label, TRUE, TRUE, 10);

    display_right_image(image_path);

    gtk_widget_show_all(results_container);
}

static void display_right_image(const gchar *image_path)
{
    if (!right_image_display || !GTK_IS_IMAGE(right_image_display))
    {
        g_print("ERROR: right_image_display is not a valid GtkImage\n");
        return;
    }

    GError *error = NULL;
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(image_path, &error);
    if (error != NULL)
    {
        g_print("ERROR: Failed to load image %s: %s\n", image_path, error->message);
        g_error_free(error);
        return;
    }
    if (pixbuf)
    {
        int container_width = 800;
        int container_height = 600;
        int width = gdk_pixbuf_get_width(pixbuf);
        int height = gdk_pixbuf_get_height(pixbuf);

        double scale_x = (double)container_width / width;
        double scale_y = (double)container_height / height;
        double scale = MIN(scale_x, scale_y);

        int new_width = (int)(width * scale);
        int new_height = (int)(height * scale);

        GdkPixbuf *scaled_pixbuf = gdk_pixbuf_scale_simple(
            pixbuf, new_width, new_height, GDK_INTERP_BILINEAR);
        gtk_image_set_from_pixbuf(GTK_IMAGE(right_image_display),
                                  scaled_pixbuf);
        g_object_unref(scaled_pixbuf);

        g_object_unref(pixbuf);
    }
}

static void process_callback(GtkWidget *widget, gpointer data)
{
    if (!current_image_path)
    {
        return;
    }

    if (is_processing)
    {
        return;
    }

    gtk_widget_set_sensitive(process_button, FALSE);

    animate_image_transition();

    process_both_modes(current_image_path);
}

static void create_processing_step_button(const char *step_name,
                                          const char *filename)
{
    GtkWidget *current_page = gtk_notebook_get_nth_page(
        GTK_NOTEBOOK(notebook), current_processing_mode);
    GtkWidget *container = tab_containers[current_processing_mode];
    GtkWidget *button_box = NULL;

    if (container == NULL)
    {
        container = gtk_scrolled_window_new(NULL, NULL);
        gtk_widget_set_size_request(container, -1, 250);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(container),
                                       GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        gtk_style_context_add_class(gtk_widget_get_style_context(container),
                                    "processing-scroller");

        gtk_box_pack_start(GTK_BOX(current_page), container, TRUE, TRUE, 5);
        tab_containers[current_processing_mode] = container;

        button_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(button_box), 5);
        gtk_style_context_add_class(gtk_widget_get_style_context(button_box),
                                    "processing-box");
        gtk_container_add(GTK_CONTAINER(container), button_box);
        gtk_widget_show_all(container);
    }
    else
    {
        GtkWidget *viewport = gtk_bin_get_child(GTK_BIN(container));
        if (viewport && GTK_IS_BIN(viewport))
        {
            button_box = gtk_bin_get_child(GTK_BIN(viewport));
        }
        if (!button_box)
        {
            button_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
            gtk_container_set_border_width(GTK_CONTAINER(button_box), 5);
            gtk_style_context_add_class(
                gtk_widget_get_style_context(button_box), "processing-box");
            gtk_container_add(GTK_CONTAINER(container), button_box);
        }
    }

    GtkWidget *button = gtk_button_new_with_label(step_name);
    gtk_widget_set_tooltip_text(button, filename);
    gtk_button_set_relief(GTK_BUTTON(button), GTK_RELIEF_NORMAL);
    gtk_widget_set_margin_start(button, 5);
    gtk_widget_set_margin_end(button, 5);
    gtk_style_context_add_class(gtk_widget_get_style_context(button),
                                "processing-step-button");

    g_signal_connect(button, "clicked", G_CALLBACK(processing_step_callback),
                     (gpointer)filename);
    gtk_box_pack_start(GTK_BOX(button_box), button, FALSE, FALSE, 0);
    gtk_widget_show(button);

    ButtonInfo *info = g_new(ButtonInfo, 1);
    info->step_name = g_strdup(step_name);
    info->filename = g_strdup(filename);
    tab_buttons[current_processing_mode] =
        g_list_append(tab_buttons[current_processing_mode], info);

    if (selected_processing_button == NULL)
    {
        processing_step_callback(button, (gpointer)filename);
    }
}

static void create_preview_button(const char *button_text, const char *filename)
{
    int saved_mode = current_processing_mode;
    current_processing_mode = 2; // Preview tab is index 2

    GtkWidget *current_page = gtk_notebook_get_nth_page(
        GTK_NOTEBOOK(notebook), current_processing_mode);
    GtkWidget *container = tab_containers[current_processing_mode];
    GtkWidget *button_box = NULL;

    if (container == NULL)
    {
        container = gtk_scrolled_window_new(NULL, NULL);
        gtk_widget_set_size_request(container, -1, 250);
        gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(container),
                                       GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
        gtk_style_context_add_class(gtk_widget_get_style_context(container),
                                    "processing-scroller");

        gtk_box_pack_start(GTK_BOX(current_page), container, TRUE, TRUE, 5);
        tab_containers[current_processing_mode] = container;

        button_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        gtk_container_set_border_width(GTK_CONTAINER(button_box), 5);
        gtk_style_context_add_class(gtk_widget_get_style_context(button_box),
                                    "processing-box");
        gtk_container_add(GTK_CONTAINER(container), button_box);
        gtk_widget_show_all(container);
    }
    else
    {
        GtkWidget *viewport = gtk_bin_get_child(GTK_BIN(container));
        if (viewport && GTK_IS_BIN(viewport))
        {
            button_box = gtk_bin_get_child(GTK_BIN(viewport));
        }
        if (!button_box)
        {
            button_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
            gtk_container_set_border_width(GTK_CONTAINER(button_box), 5);
            gtk_style_context_add_class(
                gtk_widget_get_style_context(button_box), "processing-box");
            gtk_container_add(GTK_CONTAINER(container), button_box);
        }
    }

    GtkWidget *button = gtk_button_new_with_label(button_text);
    gtk_widget_set_tooltip_text(button, filename);
    gtk_button_set_relief(GTK_BUTTON(button), GTK_RELIEF_NORMAL);
    gtk_widget_set_margin_start(button, 5);
    gtk_widget_set_margin_end(button, 5);
    gtk_style_context_add_class(gtk_widget_get_style_context(button),
                                "processing-step-button");

    g_signal_connect(button, "clicked", G_CALLBACK(processing_step_callback),
                     (gpointer)filename);
    gtk_box_pack_start(GTK_BOX(button_box), button, FALSE, FALSE, 0);
    gtk_widget_show(button);

    ButtonInfo *info = g_new(ButtonInfo, 1);
    info->step_name = g_strdup(button_text);
    info->filename = g_strdup(filename);
    tab_buttons[current_processing_mode] =
        g_list_append(tab_buttons[current_processing_mode], info);

    g_signal_connect(button, "clicked", G_CALLBACK(processing_step_callback),
                     (gpointer)info->filename);

    current_processing_mode = saved_mode;
}

static void processing_step_callback(GtkWidget *widget, gpointer data)
{
    const gchar *filename = (const gchar *)data;

    if (selected_processing_button != NULL &&
        GTK_IS_WIDGET(selected_processing_button))
    {
        GtkStyleContext *prev_ctx =
            gtk_widget_get_style_context(selected_processing_button);
        gtk_style_context_remove_class(prev_ctx, "selected");
    }

    GtkStyleContext *ctx = gtk_widget_get_style_context(widget);
    gtk_style_context_add_class(ctx, "selected");
    selected_processing_button = widget;

    GList *children =
        gtk_container_get_children(GTK_CONTAINER(results_container));
    for (GList *iter = children; iter != NULL; iter = iter->next)
    {
        gtk_widget_destroy(GTK_WIDGET(iter->data));
    }
    g_list_free(children);

    if (g_file_test(filename, G_FILE_TEST_EXISTS))
    {
        GtkWidget *right_image_frame = gtk_frame_new(NULL);
        gtk_frame_set_shadow_type(GTK_FRAME(right_image_frame), GTK_SHADOW_IN);
        gtk_style_context_add_class(
            gtk_widget_get_style_context(right_image_frame), "card-frame");
        gtk_box_pack_start(GTK_BOX(results_container), right_image_frame, FALSE,
                           FALSE, 0);

        right_image_display = gtk_image_new();
        gtk_widget_set_size_request(right_image_display, 800, 600);
        gtk_container_add(GTK_CONTAINER(right_image_frame),
                          right_image_display);

        GtkWidget *section_title = gtk_label_new(NULL);
        gtk_label_set_markup(
            GTK_LABEL(section_title),
            "<b><span size='x-large'>Processing Step</span></b>");
        gtk_style_context_add_class(gtk_widget_get_style_context(section_title),
                                    "section-title");
        gtk_box_pack_start(GTK_BOX(results_container), section_title, FALSE,
                           FALSE, 10);

        const gchar *step_name = gtk_button_get_label(GTK_BUTTON(widget));
        GtkWidget *step_label = gtk_label_new(NULL);
        gtk_label_set_line_wrap(GTK_LABEL(step_label), TRUE);
        char markup[256];
        sprintf(markup, "<b><span size='large'>%s</span></b>", step_name);
        gtk_label_set_markup(GTK_LABEL(step_label), markup);
        gtk_label_set_justify(GTK_LABEL(step_label), GTK_JUSTIFY_CENTER);
        gtk_style_context_add_class(gtk_widget_get_style_context(step_label),
                                    "step-title");
        gtk_box_pack_start(GTK_BOX(results_container), step_label, FALSE, FALSE,
                           10);

        display_right_image(filename);

        gtk_widget_show_all(results_container);
    }
    else
    {
        GtkWidget *error_label =
            gtk_label_new("Processing step image not found.\nMake sure "
                          "processing has completed.");
        gtk_label_set_line_wrap(GTK_LABEL(error_label), TRUE);
        gtk_box_pack_start(GTK_BOX(results_container), error_label, FALSE,
                           FALSE, 0);
        gtk_widget_show(error_label);

        g_print("Warning: %s not found\n", filename);
    }
}

static void animate_image_transition(void)
{
    is_processing = TRUE;

    gtk_paned_set_position(GTK_PANED(paned), 280);

    gtk_widget_queue_draw(main_window);

    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }
}

static void load_dark_theme_css(void)
{
    GtkCssProvider *provider = gtk_css_provider_new();
    GError *error = NULL;
    gchar *css_path = g_build_filename(g_get_current_dir(), "assets", "ui",
                                       "theme.css", NULL);
    gtk_css_provider_load_from_path(provider, css_path, &error);
    if (error != NULL)
    {
        g_warning("Failed to load CSS from %s: %s", css_path, error->message);
        g_error_free(error);
    }

    GdkScreen *screen = gdk_screen_get_default();
    gtk_style_context_add_provider_for_screen(
        screen, GTK_STYLE_PROVIDER(provider), GTK_STYLE_PROVIDER_PRIORITY_USER);
    g_object_unref(provider);
    g_free(css_path);

    GtkSettings *settings = gtk_settings_get_default();
    g_object_set(settings, "gtk-application-prefer-dark-theme", TRUE, NULL);
}

static void show_welcome_panel(void)
{
    GList *children =
        gtk_container_get_children(GTK_CONTAINER(results_container));
    for (GList *iter = children; iter != NULL; iter = iter->next)
    {
        gtk_widget_destroy(GTK_WIDGET(iter->data));
    }
    g_list_free(children);

    GtkWidget *welcome_label = gtk_label_new(NULL);
    gtk_label_set_markup(GTK_LABEL(welcome_label),
                         "Load an image using the button above.");
    gtk_label_set_line_wrap(GTK_LABEL(welcome_label), TRUE);
    gtk_label_set_justify(GTK_LABEL(welcome_label), GTK_JUSTIFY_CENTER);
    gtk_style_context_add_class(gtk_widget_get_style_context(welcome_label),
                                "welcome-fallback");
    gtk_box_pack_start(GTK_BOX(results_container), welcome_label, TRUE, TRUE,
                       20);

    gtk_widget_show_all(results_container);
}


static void process_both_modes(const gchar *image_path)
{
    GList *children =
        gtk_container_get_children(GTK_CONTAINER(results_container));
    for (GList *iter = children; iter != NULL; iter = iter->next)
    {
        gtk_widget_destroy(GTK_WIDGET(iter->data));
    }
    g_list_free(children);

    GtkWidget *processing_label =
        gtk_label_new("Processing image (Grid Detection + Word Detection)...");
    gtk_label_set_line_wrap(GTK_LABEL(processing_label), TRUE);
    gtk_box_pack_start(GTK_BOX(results_container), processing_label, FALSE,
                       FALSE, 0);
    gtk_widget_show(processing_label);

    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }

    int grid_result = 0;
    int word_result = 0;

    printf("Processing Grid Detection...\n");
    int saved_mode = current_processing_mode;
    current_processing_mode = 0;

    gtk_label_set_text(GTK_LABEL(processing_label),
                       "Processing Grid Detection...");
    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }

    int grid_num_rows, grid_num_cols, grid_num_cells;
    int grid_crop_offset_x, grid_crop_offset_y;
    Rect *grid_cell_bounding_boxes;
    grid_result = process_wordsearch_image(image_path, create_processing_step_button,
                                         &grid_num_rows, &grid_num_cols, &grid_cell_bounding_boxes, &grid_num_cells,
                                         &grid_crop_offset_x, &grid_crop_offset_y);
    printf("Grid Detection completed with result: %d (%d x %d grid, %d cells)\n", grid_result, grid_num_rows, grid_num_cols, grid_num_cells);

    current_processing_mode = 1;

    gtk_label_set_text(GTK_LABEL(processing_label),
                       "Processing Word Detection...");
    while (gtk_events_pending())
    {
        gtk_main_iteration();
    }

    word_result =
        process_word_detection(image_path, create_processing_step_button);
    printf("Word Detection completed with result: %d\n", word_result);

    current_processing_mode = saved_mode;

    gtk_widget_destroy(processing_label);

    if (grid_result != 0 || word_result != 0)
    {
        GList *error_children =
            gtk_container_get_children(GTK_CONTAINER(results_container));
        for (GList *iter = error_children; iter != NULL; iter = iter->next)
        {
            gtk_widget_destroy(GTK_WIDGET(iter->data));
        }
        g_list_free(error_children);

        char error_msg[256];
        if (grid_result != 0 && word_result != 0)
        {
            sprintf(error_msg,
                    "Both Grid Detection and Word Detection failed.\nGrid "
                    "result: %d\nWord result: %d\nCheck console for details.",
                    grid_result, word_result);
        }
        else if (grid_result != 0)
        {
            sprintf(error_msg,
                    "Grid Detection failed (result: %d), but Word Detection "
                    "succeeded.\nCheck console for details.",
                    grid_result);
        }
        else
        {
            sprintf(error_msg,
                    "Word Detection failed (result: %d), but Grid Detection "
                    "succeeded.\nCheck console for details.",
                    word_result);
        }

        GtkWidget *error_label = gtk_label_new(error_msg);
        gtk_label_set_line_wrap(GTK_LABEL(error_label), TRUE);
        gtk_box_pack_start(GTK_BOX(results_container), error_label, FALSE,
                           FALSE, 0);
        gtk_widget_show(error_label);
    }
    else
    {
        tab_processed[0] = TRUE;
        tab_processed[1] = TRUE;
        tab_processed[2] = TRUE;

        gtk_label_set_text(GTK_LABEL(processing_label),
                           "Loading CNN model and processing grid...");
        while (gtk_events_pending())
        {
            gtk_main_iteration();
        }

        CNN* model = cnn_create();
        cnn_load_weights(model, 12);
        cnn_eval(model);

        Grid* grid = create_grid(grid_num_rows, grid_num_cols, "cells", model);
        char* grid_str = grid_to_string(grid);

        WordsArray* words_array = create_words_array("words", model);

        printf("Grid:\n%s\n", grid_str);
        free(grid_str);

        WordMatch** word_matches = (WordMatch**)malloc(sizeof(WordMatch*) * words_array->count);
        int num_matches = 0;

        for(int i = 0; i < words_array->count; i++){
            Word* word = &words_array->words[i];
            char* word_str = word_to_string(word);
            if (word_str) {
                WordMatch* word_match = find_best_word_match(grid, word, word_str);
                if (word_match) {
                    printf("Found word '%s' at (%d,%d) going %s (score: %.3f)\n",
                           word_match->word_str,
                           word_match->start_pos.row,
                           word_match->start_pos.col,
                           word_match->direction,
                           word_match->log_prob_score);
                    word_matches[num_matches++] = word_match;
                }
                free(word_str);
            }
        }

        if (num_matches > 0) {
            char output_path[256];
            const char *basename = strrchr(image_path, '/');
            if (basename == NULL) {
                basename = strrchr(image_path, '\\');
            }
            if (basename == NULL) {
                basename = image_path;
            } else {
                basename++;
            }
            char *dot_pos = strrchr(basename, '.');
            if (dot_pos != NULL) {
                sprintf(output_path, "solved/%.*s_solved%s", (int)(dot_pos - basename), basename, dot_pos);
            } else {
                sprintf(output_path, "solved/%s_solved", basename);
            }

            draw_solved_words(image_path, word_matches, num_matches, grid_num_rows, grid_num_cols,
                              grid_cell_bounding_boxes, grid_crop_offset_x, grid_crop_offset_y, output_path);

            create_preview_button("Solved Image", output_path);

            for (int i = 0; i < num_matches; i++) {
                free_word_match(word_matches[i]);
            }

            char success_msg[512];
            sprintf(success_msg,
                    "Image processed successfully!\n\n");

            GtkWidget *success_label = gtk_label_new(success_msg);
            gtk_label_set_line_wrap(GTK_LABEL(success_label), TRUE);
            gtk_box_pack_start(GTK_BOX(results_container), success_label, FALSE,
                               FALSE, 0);
            gtk_widget_show(success_label);
        } else {
            GtkWidget *success_label = gtk_label_new("Image processed successfully!\n\n");
            gtk_label_set_line_wrap(GTK_LABEL(success_label), TRUE);
            gtk_box_pack_start(GTK_BOX(results_container), success_label, FALSE,
                               FALSE, 0);
            gtk_widget_show(success_label);
        }

        free(word_matches);
        if (grid_cell_bounding_boxes) free(grid_cell_bounding_boxes);
        FreeGrid(grid);
        FreeWordsArray(words_array);
        cnn_free(model);
    }

    is_processing = FALSE;
    gtk_widget_set_sensitive(process_button, TRUE);
}
