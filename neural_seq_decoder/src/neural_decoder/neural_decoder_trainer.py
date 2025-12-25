import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder,LSTMDecoder, TransformerDecoder
from .dataset import SpeechDataset
from .augmentations import GaussianSmoothing, TimeMasking
from .losses import FocalCTCLoss

def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"
    # device = "cpu"
    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
    # * Baseline GRU Model * #    
    # model = GRUDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits"],
    #     layer_dim=args["nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)

    # # /* baseline settings */
    # # define loss function and optimizer
    # loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=args["lrStart"],
    #     betas=(0.9, 0.999),
    #     eps=0.1,
    #     weight_decay=args["l2_decay"],
    # )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer,
    #     start_factor=1.0,
    #     end_factor=args["lrEnd"] / args["lrStart"],
    #     total_iters=args["nBatch"],
    # )
    
    # --- update: LSTM  Model Initialization --- # 
    # Switch from GRUDecoder to LSTMDecoder
    # model = LSTMDecoder(
    #     neural_dim=args["nInputFeatures"],
    #     n_classes=args["nClasses"],
    #     hidden_dim=args["nUnits"],
    #     layer_dim=args["nLayers"],
    #     nDays=len(loadedData["train"]),
    #     dropout=args["dropout"],
    #     device=device,
    #     strideLen=args["strideLen"],
    #     kernelLen=args["kernelLen"],
    #     gaussianSmoothWidth=args["gaussianSmoothWidth"],
    #     bidirectional=args["bidirectional"],
    # ).to(device)
    # -------------------------------------

    # --- update Initialize Transformer Model ---
    model = TransformerDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],  # Controls the depth of the Transformer
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=False # Must be False
    ).to(device)
    # ---------------------------------------------



    # --- [EXPERIMENT 3: Initialize Augmentation] ---
    # Initialize TimeMasking with 50% probability and a max mask length of 30 time bins (~0.6 seconds)
    time_masker = TimeMasking(p=0.5, max_mask_len=30).to(device)
    # -----------------------------------------------

    # CTC loss update: use FocalCTCLoss, gamma: 1.0 or 2.0
    loss_ctc = FocalCTCLoss(blank=0, gamma=2.0, reduction="mean", zero_infinity=True)

    # loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # /* update optimizer and scheduler */
    # 1. Upgrade optimizer to AdamW (as suggested in project directions for better generalization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lrStart"],    # 0.002
        betas=(0.9, 0.999),
        eps=1e-8,              # Reduce epsilon to PyTorch default (1e-8) for numerical stability
        weight_decay=args["l2_decay"],
    )

    # # 2. Update scheduler to CosineAnnealingLR
    # # Old code: scheduler = torch.optim.lr_scheduler.LinearLR(...)
    
    # # New code:
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=args["nBatch"],  # Total number of steps in the training cycle (10,000)
    #     eta_min=1e-6           # Key Point! Decays LR to a very small value (0.000001) at the end, unlike the baseline's 0.02
    # )


    # --- [New Scheduler with Warmup] ---
    
    # 1. Warmup 阶段: 前 500 个 Batch，学习率从 LR*0.01 线性增加到 LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=1000
    )
    
    # 2. Cosine 阶段: 500 个 Batch 之后，开始余弦退火
    # 注意 T_max 要减去 warmup 的步数
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args["nBatch"] - 1000, 
        eta_min=1e-6
    )
    
    # 3. 串联起来: 自动切换
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[1000]
    )
    # -----------------------------------

    # train the model
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Noise augmentation is faster on GPU
        if args["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        # --- update [EXPERIMENT 3: Apply Time Masking] ---
        # Apply masking only if the model is in training mode
        if model.training:
            X = time_masker(X)
        # ------------------------------------------
        # Compute prediction error
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # # update
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # #####
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()