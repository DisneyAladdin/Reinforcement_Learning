from keras.models import Sequential
from keras.layers import Dense, Activation

# モデルの作成
model = Sequential()

# モデルにレイヤーを積み上げていく
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 訓練プロセスの定義
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# 訓練の実行
# (x_train, y_trainはNumpy行列の学習データ)
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 予測の実行
classes = model.predict(x_test, batch_size=128)
