# ColorizeImage
 Тема раскраски ЧБ фотографий меня заинтересовала, потому что я сам немного занимаюсь пленочной фотографией, и в целом мне стало интересно, как выглядели улицы городов 70-100 лет назад в цвете. Поэтому для итогового проекта по курсу "Введение в Искуственный Интелект" я решил написать программу на основе нейронной сети, которая будет раскрашивать фотографии.

Писал я на языке Python, в облачной среде разработки __GoogleColab__, а в качестве библиотеки для работы с нейросетями использовал Keras. Вот [ссылка](https://colab.research.google.com/drive/1dEFVyyml8uKbmKvZvf9PruGpkG0Hyqws?usp=sharing) на ноутбук.
## Представление изображения
Все знают, что изображения представляются в формате RGB, и сначала я думал работать с RGB представлением, но в таком случае пришлось бы предсказывать 3 слоя по черно-белому изображению. Для задачи колоризации гораздо лучше использовать цветовое пространство LAB.
В этом цветовом пространстве в качестве одного из слоев используется lightness слой, то есть “светлота” картинки.

Этот слой как раз выглядит как ЧБ представление, поэтому, работая в цветовом пространстве LAB нужно будет предсказывать только слои A и B.
Слой А отвечает за положение цвета от зеленого до красного, а B - от синего до желтого. 

Также ученые доказали, что только 6 процентов рецепторов сетчатки глаза отвечают за определение цвета, а остальные - за определение яркости. Поэтому невозможно было отказаться от выбора цветового пространства LAB.
## Датасет
Далее, расскажу про датасет и возникшие проблемы. 

У меня сразу не было цели, чтобы нейронная сеть хорошо обобщала данные и раскрашивала любые фотографии. Я решил сфокусироваться на фото переулков, домов, кварталов. В открытом доступе я не нашел большого количества данных, поэтому датасет пришлось делать самому.

Сначала я просто скачал около *четырех тысяч фотографий* улиц родного города, но, ничего хорошего из этого не вышло, так как данные были плохо распределены, да и в целом было много разного мусора и шума. После неудачной попытки с помощью утилиты [Yandex-Grabber](http://ufahameleon.ru/soft.aspx?id=2) я скачивал большое количество разных фотографий, среди которых были фото деревьев, неба, улиц, советской архитектуры и трамваев. Трамваи мне захотелось добавить, потому что на ретро-снимках я видел их достаточно часто.

Размер обучающих данных я выбрал 128x128 пикселей, так как при большей размерности обучение длилось бы достаточно долго. 

Пример картинок из датасета:

![](https://github.com/IlyaKuprik/ColorizeImage/blob/main/images/train_example.jpg)

## Архитектура сети
Для проекта я использовал архитектуру сети из [данной статьи](https://habr.com/ru/company/nix/blog/342388/) с хабра. Это сверточная нейронная сеть, в которой 12 сверточных слоев, 3 из которых со _stride_ 2. Последние слои используются вместо __MaxPooling__, так как при макс-пулинге теряется пространственная структура изображения. Хотелось раскрашивать изображения, не уменьшая их размер, поэтому в модели присутствуют слои повышения дискретизации __UpSampling__.

Вот как выглядит сама модель:

```
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
```

## Обучение
Для увелечения количества обучающих данных был написан генератор данных, который немного поворачивает изображения, отражает, приближает. Так же эксперементальным путем было установлено, что *batch_size* лучше выбирать около 100. 

```
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)
batch_size = 100
epochs_size = 350 
def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
        
saver = CustomSaver()        
model.fit_generator(image_a_b_gen(batch_size),
                    callbacks=[saver, history],
                    epochs=epochs_size,
                    steps_per_epoch= len(Xtrain) // batch_size,
                    validation_data=(Xtest,Ytest))
```

Я выбрал 350 эпох для обучениия, исходя из графика точности предсказания на тестовом и валидационном множестве:

![](https://github.com/IlyaKuprik/ColorizeImage/blob/main/images/study_graphic.png)

### Главные ссылки:

[Датасет](https://drive.google.com/file/d/16810f_ik9T_3iVPwIuSeUm0bPatIkouv/view?usp=sharing)

[Ноутбук на GoogleColab](https://colab.research.google.com/drive/1dEFVyyml8uKbmKvZvf9PruGpkG0Hyqws?usp=sharing) 

[Статья, с которой была взята модель](https://habr.com/ru/company/nix/blog/342388/)
