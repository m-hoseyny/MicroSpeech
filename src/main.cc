#include "tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/tiny_conv_model_data.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "BSP_DISCO_F746NG/Drivers/BSP/STM32746G-Discovery/stm32746g_discovery_lcd.h"


int main(int argc, char* argv[]) {
  // Set up logging.
  tflite::MicroErrorReporter micro_reporter;
  tflite::ErrorReporter* reporter = &micro_reporter;

  const tflite::Model* model = ::tflite::GetModel(g_tiny_conv_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  const int tensor_arena_size = 10 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                                 tensor_arena_size);

  tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                       reporter);

  TfLiteTensor* model_input = interpreter.input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteUInt8)) {
    reporter->Report("Bad input tensor parameters in model");
    return 1;
  }

  FeatureProvider feature_provider(kFeatureElementCount,
                                   model_input->data.uint8);

  RecognizeCommands recognizer(reporter);

  int32_t previous_time = 0;

  // Initial the LCD to show the commands
  BSP_LCD_Init();
  BSP_LCD_LayerDefaultInit(LTDC_ACTIVE_LAYER, LCD_FB_START_ADDRESS);
  BSP_LCD_SelectLayer(LTDC_ACTIVE_LAYER);
  BSP_LCD_Clear(LCD_COLOR_BLACK);
  BSP_LCD_SetFont(&LCD_DEFAULT_FONT);
  BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
  BSP_LCD_SetTextColor(LCD_COLOR_DARKBLUE);

  
  while (true) {
    const int32_t current_time = LatestAudioTimestamp();
    int how_many_new_slices = 0;
    TfLiteStatus feature_status = feature_provider.PopulateFeatureData(
        reporter, previous_time, current_time, &how_many_new_slices);
    if (feature_status != kTfLiteOk) {
      reporter->Report("Feature generation failed");
      return 1;
    }
    previous_time = current_time;
    if (how_many_new_slices == 0) {
      continue;
    }

    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      reporter->Report("Invoke failed");
      return 1;
    }

    TfLiteTensor* output = interpreter.output(0);
    uint8_t top_category_score = 0;
    int top_category_index = 0;
    for (int category_index = 0; category_index < kCategoryCount; ++category_index) {
      const uint8_t category_score = output->data.uint8[category_index];
      if (category_score > top_category_score) {
        top_category_score = category_score;
        top_category_index = category_index;
      }
    }

    const char* found_command = nullptr;
    uint8_t score = 0;
    bool is_new_command = false;
    TfLiteStatus process_status = recognizer.ProcessLatestResults(output, current_time, &found_command, &score, &is_new_command);
    if (process_status != kTfLiteOk) {
      reporter->Report("RecognizeCommands::ProcessLatestResults() failed");
      return 1;
    }
    if (is_new_command) {
      reporter->Report("Heard: %s (%d)", found_command, score);
      BSP_LCD_Clear(LCD_COLOR_BLACK);
      BSP_LCD_DisplayStringAt(0, 100, (uint8_t *)found_command, CENTER_MODE);
    }
  }

  return 0;
}

