Здесь должен лежать libpsicode_rx.so (arm64-v8a).

Собрать из корня workspace (psicode-rx = Rust JNI-ядро приёмника):

  set ANDROID_NDK_HOME=%LOCALAPPDATA%\Android\Sdk\ndk\android-ndk-r27c
  cargo ndk -t arm64-v8a -o target/jniLibs build --release -p psicode-rx

Затем скопировать:
  target/jniLibs/arm64-v8a/libpsicode_rx.so  ->  этот каталог

Пока файла нет — APK собирается, но приложение стартует в "no core" режиме
(превью камеры работает, баннер "libpsicode_rx.so отсутствует").
