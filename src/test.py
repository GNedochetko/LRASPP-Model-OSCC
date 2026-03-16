import torch
from torch.utils.data import DataLoader
from .dataset import ORCADataset
from .model import get_model
import matplotlib.pyplot as plt
import numpy as np
import os

def test_model(model_path, data_dir, num_classes=2, batch_size=4, device='cuda'):
    test_dataset = ORCADataset(data_dir, subset='testing')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # modo avaliacao

    os.makedirs('resultados', exist_ok=True)
    with torch.no_grad():
        # Acumuladores de metricas agregadas
        correct_pixels = 0
        total_pixels = 0
        tp = fp = fn = tn = 0   # classe positiva (lesao)
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)  # escolhe classe com maior probabilidade

            # Atualiza metricas por lote
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
            tp  += ((preds == 1) & (masks == 1)).sum().item()
            fp  += ((preds == 1) & (masks == 0)).sum().item()
            fn  += ((preds == 0) & (masks == 1)).sum().item()
            tn  += ((preds == 0) & (masks == 0)).sum().item()

            for j in range(images.size(0)):
                image_np = images[j].cpu().permute(1,2,0).numpy()
                # Desnormaliza para visualizacao correta em [0,1]
                mean = np.array([0.485, 0.456, 0.406], dtype=image_np.dtype)
                std = np.array([0.229, 0.224, 0.225], dtype=image_np.dtype)
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0.0, 1.0)
                mask_np = masks[j].cpu().numpy()
                pred_np = preds[j].cpu().numpy()

                fig, axs = plt.subplots(1,3, figsize=(12,4))
                axs[0].imshow(image_np)
                axs[0].set_title('Imagem')
                axs[1].imshow(mask_np, cmap='gray')
                axs[1].set_title('Mascara Real')
                axs[2].imshow(pred_np, cmap='gray')
                axs[2].set_title('Predicao')
                for ax in axs:
                    ax.axis('off')

                # Salva a figura em resultados/ com nome baseado no arquivo de origem
                global_idx = i * batch_size + j
                if global_idx < len(test_dataset.images):
                    base_name = os.path.splitext(test_dataset.images[global_idx])[0]
                else:
                    base_name = f"img_{global_idx}"
                out_path = os.path.join('resultados', f"pred_{base_name}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

        # Calcula metricas agregadas ao final
        pixel_acc = correct_pixels / max(total_pixels, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        f1_score = (2 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)

        print(f"Pixel Accuracy: {pixel_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Dice: {dice:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"IoU: {iou:.4f}")

        # Salva metricas agregadas em CSV
        metrics_csv = os.path.join('resultados', 'metrics.csv')
        with open(metrics_csv, 'w', encoding='utf-8') as f:
            f.write('pixel_acc,precision,recall,specificity,dice,f1_score,iou\n')
            f.write(f"{pixel_acc:.6f},{precision:.6f},{recall:.6f},{specificity:.6f},{dice:.6f},{f1_score:.6f},{iou:.6f}\n")

    print("Teste finalizado!")


if __name__ == "__main__":
    data_dir = "data/ORCA_512x512"
    model_path = "trained_model.pth"
    test_model(model_path, data_dir, batch_size=4, device='cuda' if torch.cuda.is_available() else 'cpu')
