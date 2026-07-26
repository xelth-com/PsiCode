# JNI мост: имена native-методов и класс-держатель не переименовывать.
-keepclasseswithmembernames class com.xelth.psicode.PsiCodeCore {
    native <methods>;
}
