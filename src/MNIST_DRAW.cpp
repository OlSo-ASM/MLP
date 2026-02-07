#include <SFML/Graphics.hpp>
#include <iostream>
#include <format>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include<thread>
#include "MLP.h"

#define DEBUG

using namespace Eigen;
using namespace std;

RowVectorXf ReLU(RowVectorXf neuro);
RowVectorXf Softmax(RowVectorXf neuro);
float CrossEntropy(RowVectorXf yTrue, RowVectorXf yPred);

RowVectorXf deltaReLU(RowVectorXf neuro);
RowVectorXf deltaCrossSoft(RowVectorXf yTrue, RowVectorXf yPred);

MatrixXf dW(RowVectorXf neuro, RowVectorXf delta);

vector<size_t> sizes{ 784, 512, 256, 10 };


constexpr int HEIGHT = 500;
constexpr int WIDTH = 4*HEIGHT/3;

constexpr int frameSize = HEIGHT * 0.8;
constexpr int spaceFrame = (HEIGHT - frameSize) / 2;

constexpr int stencilSize = 5;

const int countQuad = 28;

float mapNum[28][28]{ 0.f };
sf::RectangleShape shapeMnist[28][28];

float brushStencil[stencilSize][stencilSize] =
    {{0.0, 0.1, 0.4, 0.1, 0.0 },
    { 0.1, 0.3, 0.8, 0.3, 0.1 },
    { 0.4, 0.8, 1.0, 0.8, 0.4 },
    { 0.1, 0.3, 0.8, 0.3, 0.1 },
    { 0.0, 0.1, 0.4, 0.1, 0.0}};

float raiserStencil[stencilSize][stencilSize] =
{ {0.0, 0.7, 0.8, 0.7, 0.0 },
{ 0.7, 0.9, 1.0, 0.9, 0.7 },
{ 0.8, 1.0, 1.0, 1.0, 0.8 },
{ 0.7, 0.9, 1.0, 0.9, 0.7 },
{ 0.0, 0.7, 0.8, 0.7, 0.0} };

std::vector<float> neuroInpMnist;
std::vector<float> probabilityMnist;


int main()
{

    setNbThreads(thread::hardware_concurrency());
    float lr = 0.01 / 32;
    size_t cycles = 0;
    RowVectorXf input(784);
	RowVectorXf output(10);
    MLP perceptron(sizes, ReLU, Softmax, CrossEntropy, deltaReLU, deltaCrossSoft, lr);

    perceptron.setDB_CSV("mnist_train.csv");
	for (int i = 0; i < 15; ++i)
    {
        cout << "Epoch " << i+1 << ":\n";
        perceptron.learn(0.1f, 64);
    }
   

    for (size_t i = 0; i < countQuad; ++i) {
        for (size_t j = 0; j < countQuad; ++j) {
            neuroInpMnist.push_back(mapNum[i][j]);
        }
    }
    for (size_t j = 0; j < 10; ++j) {
        probabilityMnist.push_back(0);
    }

	sf::Font font;
	if (!font.openFromFile("arial.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }
	sf::Vector2i posM = sf::Mouse::getPosition();
	bool lPressed = false;
    bool rPressed = false;
    sf::Text text(font);
	std::string info;
    sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "MLP01");
    sf::RectangleShape shape(sf::Vector2f(frameSize,frameSize));
	sf::Vector2i localPosM = sf::Mouse::getPosition();
    shape.setFillColor(sf::Color(27,27,27));
    shape.setPosition(sf::Vector2f(0, 0));
	shape.setSize(sf::Vector2f(WIDTH, HEIGHT));
    for (size_t i = 0; i < countQuad; ++i) {
		for (size_t j = 0; j < countQuad; ++j) {
            shapeMnist[i][j].setSize(sf::Vector2f(frameSize/(float)countQuad, frameSize / (float(countQuad))));
            shapeMnist[i][j].setPosition(sf::Vector2f(spaceFrame + j * (frameSize/ (float)countQuad), spaceFrame + i * (frameSize/ (float)countQuad)));
            shapeMnist[i][j].setFillColor(sf::Color::White);
        }
    }

    text.setPosition(sf::Vector2f(frameSize + spaceFrame * 2-20, spaceFrame+20));
    text.setFillColor(sf::Color(255, 255, 255));
    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (const auto* mouseMove = event->getIf<sf::Event::MouseMoved>()) {
                posM = mouseMove->position;
                if ((posM.x >= spaceFrame)&&(posM.y >= spaceFrame))
                    localPosM = (posM-sf::Vector2i(spaceFrame, spaceFrame))*countQuad/frameSize;
                else
					localPosM = sf::Vector2i(-1, -1);
            }
            if (event->is<sf::Event::KeyPressed>()) {
                if (event->getIf<sf::Event::KeyPressed>()->code == sf::Keyboard::Key::Delete) {
                    for (size_t i = 0; i < countQuad; ++i) {
                        for (size_t j = 0; j < countQuad; ++j) {
                            mapNum[i][j] = 0.f;
                            shapeMnist[i][j].setFillColor(sf::Color::White);
                        }
                    }
				}
            }
        }
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
			lPressed = 1;
        }else
			lPressed = 0;
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right)) {
            rPressed = 1;
        }
        else
            rPressed = 0;
        if ((localPosM.x >= 0) && (localPosM.x <= 27) && (localPosM.y >= 0) && (localPosM.y <= 27))
        {
            if (lPressed) {
                for (int i = localPosM.y-2; i < localPosM.y+2; ++i) {
                    for (int j = localPosM.x - 2; j < localPosM.x + 2; ++j) {
                        if ((i < 0) || (i > 27) || (j < 0) || (j > 27)) continue;
                        mapNum[i][j] = ((uint8_t)(mapNum[i][j] * 255) | (uint8_t)((brushStencil[i - (localPosM.y - 2)][j - (localPosM.x - 2)]) * 255)) / 255.f;
                        uint8_t color_tm = 255-static_cast<std::uint8_t>(mapNum[i][j] * 255.f);
                        shapeMnist[i][j].setFillColor(sf::Color(color_tm, color_tm, color_tm));
                    }
                }
            } else if (rPressed) {
                for (int i = localPosM.y - 2; i < localPosM.y + 2; ++i) {
                    for (int j = localPosM.x - 2; j < localPosM.x + 2; ++j) {
                        if ((i < 0) || (i > 27) || (j < 0) || (j > 27)) continue;
                        mapNum[i][j] = ((uint8_t)(mapNum[i][j] * 255) & ~(uint8_t)((raiserStencil[i - (localPosM.y - 2)][j - (localPosM.x - 2)]) * 255)) / 255.f;
                        uint8_t color_tm = 255-static_cast<std::uint8_t>(mapNum[i][j] * 255.f);
                        shapeMnist[i][j].setFillColor(sf::Color(color_tm, color_tm, color_tm));
                    }
                }
            }
        }

		for (size_t i = 0; i < countQuad; ++i) {
            for (size_t j = 0; j < countQuad; ++j) {
                neuroInpMnist[i*countQuad +j] = mapNum[i][j];
            }
        }

        for (size_t i = 0; i < countQuad*countQuad; ++i) {
            input(i) = neuroInpMnist[i];
        }

        perceptron.test(input);
    	output = perceptron.answer();

        for (size_t i = 0; i < 10; ++i) {
            probabilityMnist[i] = output(i)*100;
        }

		info = "0: " + std::format("{:02.0f}", probabilityMnist[0]) + "%\n"
             + "1: " + std::format("{:02.0f}", probabilityMnist[1]) + "%\n"
             + "2: " + std::format("{:02.0f}", probabilityMnist[2]) + "%\n"
             + "3: " + std::format("{:02.0f}", probabilityMnist[3]) + "%\n"
             + "4: " + std::format("{:02.0f}", probabilityMnist[4]) + "%\n"
             + "5: " + std::format("{:02.0f}", probabilityMnist[5]) + "%\n"
             + "6: " + std::format("{:02.0f}", probabilityMnist[6]) + "%\n"
             + "7: " + std::format("{:02.0f}", probabilityMnist[7]) + "%\n"
             + "8: " + std::format("{:02.0f}", probabilityMnist[8]) + "%\n"
             + "9: " + std::format("{:02.0f}", probabilityMnist[9]) + "%\n";
		text.setString(info);


        window.clear();
        window.draw(shape);
		for (size_t i = 0; i < 28; ++i) {
            for(size_t j = 0; j < 28; ++j) {
                window.draw(shapeMnist[i][j]);
            }
        }
        window.draw(text);
        window.display();
    }
}

RowVectorXf ReLU(RowVectorXf neuro) {
    return neuro.array().max(0.0f);
}

RowVectorXf Softmax(RowVectorXf neuro) {
    float maxVal = neuro.maxCoeff();
    RowVectorXf expVec = (neuro.array() - maxVal).exp();
    return expVec / expVec.sum();
}

float CrossEntropy(RowVectorXf yTrue, RowVectorXf yPred) {
    yPred = yPred.array().max(1e-15);
    return -(yTrue.array() * yPred.array().log()).sum();
}

RowVectorXf deltaReLU(RowVectorXf neuro) {
    return (neuro.array() > 0).cast<float>();
}

RowVectorXf deltaCrossSoft(RowVectorXf yTrue, RowVectorXf yPred) {
    return yPred - yTrue;
}

MatrixXf dW(RowVectorXf neuro, RowVectorXf delta) {
    return neuro.transpose() * delta;
}