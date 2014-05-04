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

#include <vector>

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
#define FRAMES_PER_BUFFER (2048)
#define PORT 7681

static int gNumNoInputs = 0;
/* This routine will be called by the PortAudio engine when audio is needed.
** It may be called at interrupt level on some machines so don't do anything
** that could mess up the system like calling malloc() or free().
*/
static fftwf_complex left_out[FRAMES_PER_BUFFER], right_out[FRAMES_PER_BUFFER];
static fftwf_plan lp, rp;

static int fftwCallback(const void *inputBuffer, void *outputBuffer,
                        unsigned long framesPerBuffer,
                        const PaStreamCallbackTimeInfo *timeInfo,
                        PaStreamCallbackFlags statusFlags, void *userData) {
  float **input_ptr_ary = (float **)inputBuffer;
  float *left_in = input_ptr_ary[0];
  float *right_in = input_ptr_ary[1];

  if (lp == NULL && rp == NULL) {
    lp = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, left_in, left_out, FFTW_MEASURE);
    rp = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER, right_in, right_out, FFTW_MEASURE);
  }

  (void)timeInfo; /* Prevent unused variable warnings. */
  (void)statusFlags;
  (void)userData;
  (void)outputBuffer;

  if (inputBuffer == NULL) {
    gNumNoInputs += 1;
  }

  /* Hanning window function */
  for (uint i = 0; i < framesPerBuffer; i++) {
    double multiplier = 0.5 * (1 - cos(2 * M_PI * i / (framesPerBuffer - 1)));
    left_in[i] = multiplier * (left_in[i] + 1.0);
    right_in[i] = multiplier * (right_in[i] + 1.0);
  }

  fftwf_execute(lp);
  fftwf_execute(rp);

  // second half of bins are useless
//  for (uint i = 0; i < (framesPerBuffer / 2) - 1; i++) {
//    printf("%-5u Hz: \tL: %f + i%f, \tR: %-5f + i%f\n",
//           i * SAMPLE_RATE / FRAMES_PER_BUFFER, left_out[i][0], left_out[i][0],
//           right_out[i][0], right_out[i][1]);
//  }
  return paContinue;
}

void setupAudio(PaStream *stream) {
  PaStreamParameters inputParameters;
  PaError err;

  err = Pa_Initialize();
  if (err != paNoError)
    Pa_Terminate();

  /* default input device */
  inputParameters.device = Pa_GetDefaultInputDevice();

  if (inputParameters.device == paNoDevice) {
    fprintf(stderr, "Error: No default input device.\n");
    Pa_Terminate();
  }

  inputParameters.channelCount = 2; /* stereo input */
  inputParameters.sampleFormat = PA_SAMPLE_TYPE;
  inputParameters.suggestedLatency =
      Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
  inputParameters.hostApiSpecificStreamInfo = NULL;

  err = Pa_OpenStream(&stream, &inputParameters, NULL, SAMPLE_RATE,
                      FRAMES_PER_BUFFER, 0,
                      /* paClipOff, */ /* we won't output out of range samples
                                          so don't bother clipping them */
                      fftwCallback, NULL);
  if (err != paNoError)
    Pa_Terminate();

  err = Pa_StartStream(stream);
  if (err != paNoError)
    Pa_Terminate();
}

/*******************************************************************/
// int main(void) {
//  PaStream *stream = nullptr;
//  setupAudio(stream);
//
//  while (true) {
//  };
//
//  Pa_CloseStream(stream);
//  Pa_Terminate();
//  std::cout << "Quiting!" << std::endl;
//  return 0;
//}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), stream(nullptr) {
  ui->setupUi(this);
  setGeometry(400, 250, 542, 390);

  setupAudio(stream);

  setupBarChart(ui->customPlot);
  setWindowTitle("QCustomPlot: " + demoName);
  statusBar()->clearMessage();
  ui->customPlot->replot();
  // setupPlayground(ui->customPlot);
  // 0: setupRealtimeDataDemo(ui->customPlot);
  // 1: setupBarChartDemo(ui->customPlot);
  // for making screenshots of the current demo or all demos (for website
  // screenshots):
  // QTimer::singleShot(1500, this, SLOT(allScreenShots()));
  // QTimer::singleShot(4000, this, SLOT(screenShot()));
}

MainWindow::~MainWindow() {
    Pa_CloseStream(stream);
    Pa_Terminate();
    delete ui;
}


void MainWindow::setupBarChart(QCustomPlot *customPlot) {
  demoName = "Bar Chart Demo";
  // create empty bar chart objects:
  QCPBars *fossil = new QCPBars(customPlot->xAxis, ui->customPlot->yAxis);
  customPlot->addPlottable(fossil);
//  // set names and colors:
//  QPen pen;
//  pen.setWidthF(1.2);
//  fossil->setName("Fossil fuels");
//  pen.setColor(QColor(255, 131, 0));
//  fossil->setPen(pen);
//  fossil->setBrush(QColor(255, 131, 0, 50));
  
    // prepare y axis:
  customPlot->yAxis->setRange(0, 65);
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

double average(vector<float> &v) {
  double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
  return sum / v.size();
}

void MainWindow::realtimeDataSlot() {
  double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0;

  // get bar chart
  QCPBars* fossil = static_cast<QCPBars*>(ui->customPlot->plottable(0));
  fossil->clearData(); // clear the current data

  // prepare x axis with country labels:
  QVector<double> ticks;

  // Add data:
  vector<float> tempdata;
  for (uint i = 2; i < (FRAMES_PER_BUFFER / 8) - 1; i++) {
    //ticks << (i * SAMPLE_RATE / FRAMES_PER_BUFFER);
//           right_out[i][0]);
    tempdata.push_back(abs(left_out[i][0]));
  }

  QVector<double> data;
  
  uint i =2;
  for (auto it = tempdata.begin(); !(it >= tempdata.end()); it+=16) { 
    vector<float> temp(it,it+16); 
    ticks << (i * SAMPLE_RATE / FRAMES_PER_BUFFER);
    i+=16;
    data << average(temp);
  } 
  


  fossil->setData(ticks, data);  // add the new data
  fossil->rescaleKeyAxis();      // scale the X axis
  ui->customPlot->replot();      // redraw


  // calculate frames per second:
  static double lastFpsKey;
  static int frameCount;
  ++frameCount;
  // average fps over 2 seconds
  if (key - lastFpsKey > 2) {
    ui->statusBar->showMessage(
        QString("%1 FPS").arg(frameCount / (key - lastFpsKey), 0, 'f', 0),0);
    lastFpsKey = key;
    frameCount = 0;
  }
}

void MainWindow::setupPlayground(QCustomPlot *customPlot) {
  Q_UNUSED(customPlot)
}

