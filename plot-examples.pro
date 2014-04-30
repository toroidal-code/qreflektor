QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = reflektor
TEMPLATE = app

SOURCES += main.cpp\
           mainwindow.cpp \
           qcustomplot.cpp

HEADERS  += mainwindow.h \
            qcustomplot.h

FORMS    += mainwindow.ui

LIBS += -L/usr/local/lib -lportaudio -lm -lfftw3f

CONFIG += c++11
