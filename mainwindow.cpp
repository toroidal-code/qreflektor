/***************************************************************************
**                                                                        **
**  QCustomPlot, an easy to use, modern plotting widget for Qt            **
**  Copyright (C) 2011, 2012, 2013, 2014 Emanuel Eichhammer               **
**                                                                        **
**  This program is free software: you can redistribute it and/or modify  **
**  it under the terms of the GNU General Public License as published by  **
**  the Free Software Foundation, either version 3 of the License, or     **
**  (at your option) any later version.                                   **
**                                                                        **
**  This program is distributed in the hope that it will be useful,       **
**  but WITHOUT ANY WARRANTY; without even the implied warranty of        **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         **
**  GNU General Public License for more details.                          **
**                                                                        **
**  You should have received a copy of the GNU General Public License     **
**  along with this program.  If not, see http://www.gnu.org/licenses/.   **
**                                                                        **
****************************************************************************
**           Author: Emanuel Eichhammer                                   **
**  Website/Contact: http://www.qcustomplot.com/                          **
**             Date: 07.04.14                                             **
**          Version: 1.2.1                                                **
****************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex.h>

#include <fftw3.h>
#include <string>
#include <iostream>
#include <sys/time.h>
#include <portaudio.h>
#include <numeric>

#include <vector>
#include <array>
#include <thread>
#include <mutex>

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include <QDesktopWidget>
#include <QScreen>
#include <QMessageBox>
#include <QMetaEnum>

#define NUM_BINS 6

using namespace std;

/*
** Note that many of the older ISA sound cards on PCs do NOT support
** full duplex audio (simultaneous record and playback).
** And some only support full duplex at lower sample rates.
*/
#define SAMPLE_RATE (44100)
#define PA_SAMPLE_TYPE paFloat32 | paNonInterleaved;
#define FRAMES_PER_BUFFER (4096)
#define PORT 7681

double gInOutScaler = 1.0;
#define CONVERT_IN_TO_OUT(in) ((float)((in) * gInOutScaler))

/* This routine will be called by the PortAudio engine when audio is needed.
** It may be called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static fftwf_complex left_out[FRAMES_PER_BUFFER], right_out[FRAMES_PER_BUFFER];
static fftwf_plan lp, rp;
static float *left_mid;
static float *right_mid;
mutex mid_mtx;
mutex out_mtx;

mutex matrix_mtx;

// 1/3 octave middle frequency array
static const float omf[] = { 15.6, 31.3, 62.5, 125,  250,  500,
                             1000, 2000, 4000, 8000, 16000 };

// 1/3 octave middle frequency array
static const float tomf[] = { 15.6,    19.7,   24.8,   31.3,   39.4,    49.6,
                              62.5,    78.7,   99.2,   125.0,  157.5,   198.4,
                              250,     315,    396.9,  500.0,  630,     793.7,
                              1000.0,  1259.9, 1587.4, 2000,   2519.8,  3174.8,
                              4000.0,  5039.7, 6349.6, 8000.0, 10079.4, 12699.2,
                              16000.0, 20158.7 };

static float lbtomf[32] = { 0 };
static float ubtomf[32] = { 0 };

template <typename T, class C> inline T average(C &c) {
  float a = 0;
  float scalar = (c.size() / 32.0);
  for (auto b : c) {
    a += b * (scalar < 1 ? 1 : scalar);
  }
  return a / c.size();
}

template <typename T, class C> inline T max(C &c) {
  T a = 0;
  for (auto b : c) {
    if (b > a) { a = b; }
  }
  return a;
}

/**
 * Calculate the gain for a given frequency.
 * based on http://www.ap.com/kb/show/480
 * band: bandwidth designator (1 for full octave, 3 for 1/3-octave,… etc.)
 * freq: frequency
 * fm: the mid-band frequency of the 1/b-octave filter
 */
template <typename T> inline T calculate_gain(T band, T freq, T fm) {
  return sqrt(1.0 /
              (1.0 + pow(((freq / fm) - (fm / freq)) * (1.507 * band), 6.0)));
}

inline float upper_freq_bound(float mid, float band) {
  return mid * pow(sqrt(2), 1.0 / band);
}

inline float lower_freq_bound(float mid, float band) {
  return mid / pow(sqrt(2), 1.0 / band);
}

void populate_bound_arrays() {
  for (uint i = 0; i < 32; i++) {
    lbtomf[i] = lower_freq_bound(tomf[i], 3);
    ubtomf[i] = upper_freq_bound(tomf[i], 3);
  }
}

static unordered_map<int, int> freq_octave_map;

// TODO: make this some sort of tree thing
// Instead of an O(N) lookup
int get_octave_bin(float freq) {
  auto got = mymap.find(freq);
  if ( got != mymap.end() ) {  // we found a freq -> octave map
    return *got;               // so return the octave
  } else {
    for (int i = 0; i < 3 * 11; i++) {
      if (lbtomf[i] <= freq && freq < ubtomf[i]) {
        freq_octave_map[freq] = i;
        return i;
      }
    }
  }
  return -1;
}

static void fftwProcess(const void *inputBuffer) {
  if (inputBuffer == NULL)
    return;
  float **input_ptr_ary = (float **)inputBuffer;
  float *left_in = input_ptr_ary[0];
  float *right_in = input_ptr_ary[1];

  mid_mtx.lock();
  /* Hanning window function */
  for (uint i = 0; i < FRAMES_PER_BUFFER; i++) {
    double multiplier = 0.5 * (1 - cos(2 * M_PI * i / (FRAMES_PER_BUFFER - 1)));
    left_mid[i] = multiplier * (left_in[i] + 1.0);
    right_mid[i] = multiplier * (right_in[i] + 1.0);
  }
  mid_mtx.unlock();

  out_mtx.lock();
  fftwf_execute(lp);
  fftwf_execute(rp);
  out_mtx.unlock();
}

static int copyCallback(const void *inputBuffer, void *outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo *, PaStreamCallbackFlags,
                        void *) {
  //    Copy stuff
  if (inputBuffer == NULL)
    return 0;
  memcpy(((float **)outputBuffer)[0], ((float **)inputBuffer)[0],
         framesPerBuffer * sizeof(float));
  memcpy(((float **)outputBuffer)[1], ((float **)inputBuffer)[1],
         framesPerBuffer * sizeof(float));
  // thread (fftwProcess, inputBuffer).detach();

  if (inputBuffer == NULL)
    return paContinue;
  float **input_ptr_ary = (float **)inputBuffer;
  float *left_in = input_ptr_ary[0];
  float *right_in = input_ptr_ary[1];

  /* Hanning window function */
  for (uint i = 0; i < FRAMES_PER_BUFFER; i++) {
    double multiplier = 0.5 * (1 - cos(2 * M_PI * i / (FRAMES_PER_BUFFER - 1)));
    left_mid[i] = multiplier * (left_in[i] + 1.0);
    right_mid[i] = multiplier * (right_in[i] + 1.0);
  }

  fftwf_execute(lp);
  fftwf_execute(rp);

  return paContinue;
}

void setupAudio(PaStream *stream) {
  PaStreamParameters inputParameters, outputParameters;
  PaError err;

  err = Pa_Initialize();
  if (err != paNoError)
    Pa_Terminate();

  /* default input device */
  inputParameters.device = Pa_GetDefaultInputDevice();
  outputParameters.device = Pa_GetDefaultOutputDevice();

  if (inputParameters.device == paNoDevice) {
    fprintf(stderr, "Error: No default input device.\n");
    Pa_Terminate();
  }

  inputParameters.channelCount = 2; /* stereo input */
  inputParameters.sampleFormat = PA_SAMPLE_TYPE;
  inputParameters.suggestedLatency =
      Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
  inputParameters.hostApiSpecificStreamInfo = NULL;

  outputParameters.channelCount = 2; /* stereo output */
  outputParameters.sampleFormat = PA_SAMPLE_TYPE;
  outputParameters.suggestedLatency =
      Pa_GetDeviceInfo(outputParameters.device)->defaultLowInputLatency;
  outputParameters.hostApiSpecificStreamInfo = NULL;

  err = Pa_OpenStream(&stream, &inputParameters, &outputParameters, SAMPLE_RATE,
                      FRAMES_PER_BUFFER, 0,
                      /* paClipOff, */ /* we won't output out of range samples
                                          so don't bother clipping them */
                      copyCallback, NULL);

  if (err != paNoError)
    Pa_Terminate();

  err = Pa_StartStream(stream);
  if (err != paNoError)
    Pa_Terminate();
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), stream(nullptr) {
  left_mid = (float *)fftwf_malloc(sizeof(float) * FRAMES_PER_BUFFER);
  right_mid = (float *)fftwf_malloc(sizeof(float) * FRAMES_PER_BUFFER);
  lp = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, left_mid, left_out,
                             FFTW_MEASURE);
  rp = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, right_mid, right_out,
                             FFTW_MEASURE);
  populate_bound_arrays();

  ui->setupUi(this);
  setGeometry(400, 250, 542, 390);

  setupAudio(stream);

  setupBarChart(ui->customPlot);
  setWindowTitle("QCustomPlot: " + demoName);
  statusBar()->clearMessage();
  ui->customPlot->replot();
}

MainWindow::~MainWindow() {
  fftwf_free(left_mid);
  fftwf_free(right_mid);
  Pa_CloseStream(stream);
  Pa_Terminate();
  delete ui;
}

void MainWindow::setupBarChart(QCustomPlot *customPlot) {
  demoName = "Bar Chart Demo";
  // create empty bar chart objects:
  QCPBars *fossil = new QCPBars(customPlot->xAxis, ui->customPlot->yAxis);
  customPlot->addPlottable(fossil);

  // prepare y axis:
  customPlot->yAxis->setRange(0, 85);
  customPlot->yAxis->setPadding(5); // a bit more space to the left border
  customPlot->yAxis->setLabel("DB");
  customPlot->yAxis->grid()->setSubGridVisible(true);
  QPen gridPen;
  gridPen.setStyle(Qt::SolidLine);
  gridPen.setColor(QColor(0, 0, 0, 25));
  customPlot->yAxis->grid()->setPen(gridPen);
  gridPen.setStyle(Qt::DotLine);
  customPlot->yAxis->grid()->setSubGridPen(gridPen);

  customPlot->xAxis->setRange(0, 10);
  customPlot->xAxis->setPadding(5); // a bit more space to the left border
  customPlot->xAxis->setLabel("Hz");
  customPlot->xAxis->grid()->setSubGridVisible(true);
  gridPen.setStyle(Qt::SolidLine);
  gridPen.setColor(QColor(0, 0, 0, 25));
  customPlot->xAxis->grid()->setPen(gridPen);
  gridPen.setStyle(Qt::DotLine);
  customPlot->xAxis->grid()->setSubGridPen(gridPen);

  // setup a timer that repeatedly calls MainWindow::realtimeDataSlot:
  connect(&dataTimer, SIGNAL(timeout()), this, SLOT(realtimeDataSlot()));
  dataTimer.start(0); // Interval 0 means to refresh as fast as possible
}

void MainWindow::realtimeDataSlot() {
  double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0;

  // get bar chart
  QCPBars *fossil = static_cast<QCPBars *>(ui->customPlot->plottable(0));
  fossil->clearData(); // clear the current data

  // 0 is DC freq (O Hz)
  // n/2 is nyquist freq
  array<vector<float>, 32> octave_bins;
  for (uint i = 2; i < (FRAMES_PER_BUFFER / 2) - 1; i++) {
    float freq   = (i * SAMPLE_RATE / FRAMES_PER_BUFFER);
    int   octave = get_octave_bin(freq);
    float gain   = calculate_gain<float>(3, freq, tomf[octave]);
    float val    = abs(left_out[i][0]);
    octave_bins[octave].push_back(val * gain);
  }

  // prepare x axis with labels:
  QVector<double> ticks;
  QVector<double> data;
  uint i = 0;
  for (vector<float> bin : octave_bins) {
    ticks << i++;
    double maxed = max<double>(bin);
    if (isnan(maxed)) {
      data.push_back(0);
    } else if (maxed > 80) {
      data.push_back(80);
    } else {
      data.push_back(maxed);
    }
  }

  fossil->setData(ticks, data); // add the new data
  fossil->rescaleKeyAxis();     // scale the X axis
  ui->customPlot->replot();     // redraw

  // calculate frames per second:
  static double lastFpsKey;
  static int frameCount;
  ++frameCount;
  // average fps over 2 seconds
  if (key - lastFpsKey > 2) {
    ui->statusBar->showMessage(
        QString("%1 FPS").arg(frameCount / (key - lastFpsKey), 0, 'f', 0), 0);
    lastFpsKey = key;
    frameCount = 0;
  }
}

void MainWindow::setupPlayground(QCustomPlot *customPlot) {
  Q_UNUSED(customPlot)
}
