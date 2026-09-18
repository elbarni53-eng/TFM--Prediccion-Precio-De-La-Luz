"""Red LSTM (seccion 6.2.4 de la memoria).

Dos capas LSTM apiladas de 64 unidades, dropout 0.2 entre capas, capa
lineal de salida, entrenada con perdida L1 (MAE) durante un maximo de 6
epocas con parada temprana (paciencia 2) sobre la particion de validacion.

Nota de implementacion importante: una primera version que materializaba
explicitamente todas las ventanas de 168 horas en memoria (una matriz de
aproximadamente 52 000 x 168 x 23 valores) agoto la memoria disponible. La
version de este modulo genera cada ventana de forma perezosa mediante un
`Dataset` de PyTorch que indexa una unica matriz bidimensional, reduciendo
el consumo de memoria en mas de un orden de magnitud sin alterar el
resultado del entrenamiento.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

VENTANA_HORAS = 168
UNIDADES_OCULTAS = 64
NUM_CAPAS = 2
DROPOUT = 0.2
EPOCAS_MAXIMAS = 6
PACIENCIA_PARADA_TEMPRANA = 2
BATCH_SIZE = 256


class VentanaDeslizanteDataset(Dataset):
    """Genera ventanas de `VENTANA_HORAS` horas de forma perezosa a partir
    de una matriz 2D (n_filas, n_features) ya normalizada, en lugar de
    materializar todas las ventanas de antemano.
    """

    def __init__(self, features: np.ndarray, objetivo: np.ndarray):
        assert len(features) == len(objetivo)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.objetivo = torch.as_tensor(objetivo, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features) - VENTANA_HORAS

    def __getitem__(self, idx: int):
        ventana = self.features[idx: idx + VENTANA_HORAS]
        y = self.objetivo[idx + VENTANA_HORAS]
        return ventana, y


class LSTMPrecio(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=UNIDADES_OCULTAS,
            num_layers=NUM_CAPAS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.salida = nn.Linear(UNIDADES_OCULTAS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        salida_lstm, _ = self.lstm(x)
        ultimo_paso = salida_lstm[:, -1, :]
        return self.salida(ultimo_paso).squeeze(-1)


def normalizar_con_estadisticos_train(
    train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza a media 0 y desviacion tipica 1, calculando ambos
    estadisticos EXCLUSIVAMENTE sobre `train` para no filtrar informacion
    de validacion o test hacia el preprocesado.
    """
    media = train.mean(axis=0, keepdims=True)
    desviacion = train.std(axis=0, keepdims=True)
    desviacion[desviacion == 0] = 1.0
    return (train - media) / desviacion, (val - media) / desviacion, (test - media) / desviacion


def entrenar_lstm(
    train_features: np.ndarray, train_objetivo: np.ndarray,
    val_features: np.ndarray, val_objetivo: np.ndarray,
    epocas_maximas: int = EPOCAS_MAXIMAS,
    paciencia: int = PACIENCIA_PARADA_TEMPRANA,
) -> LSTMPrecio:
    """Bucle de entrenamiento con parada temprana manual sobre el MAE de
    validacion. `epocas_maximas` y `paciencia` son parametrizables para
    poder reproducir el experimento de sobreajuste de la seccion 7.8
    (40 epocas, sin parada temprana) reutilizando esta misma funcion.
    """
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = LSTMPrecio(n_features=train_features.shape[1]).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida_fn = nn.L1Loss()

    train_loader = DataLoader(
        VentanaDeslizanteDataset(train_features, train_objetivo),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        VentanaDeslizanteDataset(val_features, val_objetivo),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    mejor_mae_val = float("inf")
    epocas_sin_mejora = 0
    mejor_estado = None

    for epoca in range(1, epocas_maximas + 1):
        modelo.train()
        for lote_x, lote_y in train_loader:
            lote_x, lote_y = lote_x.to(dispositivo), lote_y.to(dispositivo)
            optimizador.zero_grad()
            pred = modelo(lote_x)
            perdida = perdida_fn(pred, lote_y)
            perdida.backward()
            optimizador.step()

        modelo.eval()
        errores_val = []
        with torch.no_grad():
            for lote_x, lote_y in val_loader:
                lote_x, lote_y = lote_x.to(dispositivo), lote_y.to(dispositivo)
                pred = modelo(lote_x)
                errores_val.append((pred - lote_y).abs())
        mae_val = torch.cat(errores_val).mean().item()
        print(f"Epoca {epoca}/{epocas_maximas}  MAE validacion = {mae_val:.3f}")

        if mae_val < mejor_mae_val:
            mejor_mae_val = mae_val
            epocas_sin_mejora = 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= paciencia:
                print(f"Parada temprana en la epoca {epoca} (paciencia={paciencia}).")
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo


def predecir_lstm(modelo: LSTMPrecio, features: np.ndarray, objetivo: np.ndarray) -> np.ndarray:
    """Genera predicciones para todo el rango disponible. Las primeras
    `VENTANA_HORAS` observaciones de cada particion no tienen ventana
    completa y no se predicen (igual que el resto de features con lag).
    """
    dispositivo = next(modelo.parameters()).device
    dataset = VentanaDeslizanteDataset(features, objetivo)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for lote_x, _ in loader:
            lote_x = lote_x.to(dispositivo)
            predicciones.append(modelo(lote_x).cpu().numpy())
    return np.concatenate(predicciones)
