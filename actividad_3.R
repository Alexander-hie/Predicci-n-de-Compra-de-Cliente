# Instalar librerías necesarias (si no las tienes instaladas)
if (!require("data.table")) install.packages("data.table")
if (!require("caret")) install.packages("caret")
if (!require("pROC")) install.packages("pROC")
if (!require("randomForest")) install.packages("randomForest")
if (!require("rpart")) install.packages("rpart")

# Cargar las librerías
library(data.table)
library(caret)
library(pROC)
library(randomForest)
library(rpart)


set.seed(42)

n <- 10000
edad <- sample(18:70, n, replace = TRUE)
ingreso <- sample(2000:10000, n, replace = TRUE)
genero <- sample(c("Hombre", "Mujer"), n, replace = TRUE)
visitas_web <- sample(0:50, n, replace = TRUE)

compra <- as.numeric(runif(n) < (0.3 * (edad < 30) + 0.5 * (ingreso > 5000) + 0.2 * (visitas_web > 20)))

datos <- data.table(edad, ingreso, genero, visitas_web, compra)

datos[, genero := as.numeric(genero == "Hombre")]

datos[, compra := as.factor(compra)]

head(datos)

set.seed(42)
trainIndex <- createDataPartition(datos$compra, p = 0.7, list = FALSE)
entrenamiento <- datos[trainIndex]
prueba <- datos[-trainIndex]

# 1. Modelo de Regresión Logística
modelo_log <- glm(compra ~ ., data = entrenamiento, family = "binomial")
pred_log <- predict(modelo_log, prueba, type = "response")

# 2. Modelo de Árbol de Decisión
modelo_tree <- rpart(compra ~ ., data = entrenamiento, method = "class")
pred_tree <- predict(modelo_tree, prueba, type = "prob")[,2]

# 3. Modelo de Random Forest
modelo_rf <- randomForest(compra ~ ., data = entrenamiento)
pred_rf <- predict(modelo_rf, prueba, type = "prob")[,2]

# Calcular las curvas ROC
roc_log <- roc(as.numeric(prueba$compra) - 1, pred_log)
auc_log <- auc(roc_log)

roc_tree <- roc(as.numeric(prueba$compra) - 1, pred_tree)
auc_tree <- auc(roc_tree)

roc_rf <- roc(as.numeric(prueba$compra) - 1, pred_rf)
auc_rf <- auc(roc_rf)

# Mostrar los valores AUC
cat("AUC de los modelos:\n")
cat("Regresión Logística:", auc_log, "\n")
cat("Árbol de Decisión:", auc_tree, "\n")
cat("Random Forest:", auc_rf, "\n")

# Graficar las curvas ROC
plot(roc_log, col = "blue", lwd = 2, main = "Comparación de Curvas ROC")
lines(roc_tree, col = "red", lwd = 2)
lines(roc_rf, col = "green", lwd = 2)
legend("bottomright", legend = c("Regresión Logística", "Árbol de Decisión", "Random Forest"),
       col = c("blue", "red", "green"), lwd = 2)
