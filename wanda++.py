def compute_rgs_score(block: nn.Module, inps: List[torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    sq_grad = {name: torch.zeros_like(param) for name, param in block.named_parameters()}

    for inp in inps:
        loss = torch.norm(block(inp))
        loss.backward(retain_graph=True)

        with torch.no_grad():
            for name, param in block.named_parameters():
                if param.grad is not None:
                    sq_grad[name] += param.grad.detach() ** 2
        block.zero_grad()

    inps_len = len(inps)
    for key in sq_grad:
        sq_grad[key] = torch.sqrt(sq_grad[key] / inps_len)

    score_dict = {}
    # 遍历当前块中所有可学习的参数
    for name, param in block.named_parameters():
        if param.requires_grad:
            # 获取当前层的输入X，inps是多个样本的集合，get_signal从每个样本中提取出这层的输入，然后stack合并成一个张量
            layer_inps = torch.stack(get_signal(inps, name))
            norm_term = layer_inps.norm(p=2, dim=0)
            score = param.data.abs() * (alpha * sq_grad[name] + norm_term)
            # 把权重的剪枝打分保存起来，后续用于排序和剪枝
            score_dict[name] = score

    return score_dict

# ========== Algorithm 1: Wanda++ Pruning ==========
def wanda_plus_plus_pruning(model: nn.Module, input_sets: List[List[torch.Tensor]], alpha: float = 0.5, K: int = 5):
    """
    Args:
        model: Transformer with decoder blocks
        input_sets: List[List[Tensor]] — X^l for each decoder block l
        alpha: scaling factor for score
        K: RO optimization rounds
    Returns:
        pruned model
    """
    decoder_blocks = [module for module in model.modules() if isinstance(module, nn.TransformerDecoderLayer)]

    for l, block in enumerate(decoder_blocks):
        X_l = input_sets[l]

        # Step 1: RGS loss & gradient collection
        rgs_score = compute_rgs_score(block, X_l, alpha)

        for k in range(K):
            # Step 2: Select RO samples (you can do random sampling here) ，eg select 2 samples
            X_hat_l = random.sample(X_l, min(len(X_l), 2))  

            # Step 3: Pruning by RGS
            for name, param in block.named_parameters():
                if name in rgs_score:
                    # Prune lowest X% scores (e.g. 30%)
                    score = rgs_score[name]
                    threshold = torch.quantile(score, 0.3)
                    mask = (score > threshold).float()
                    param.data *= mask  # Zero out pruned weights

            # Step 4: Regional Optimization (fine-tune with small batch)
            # 复制未剪枝的 block 作为 teacher（冻结参数）
            block_orig = deepcopy(block)
            block_orig.eval()
            for p in block_orig.parameters():
                p.requires_grad = False

            # 启用当前剪枝后的 block（作为 student）
            block.train()
            optimizer = torch.optim.RMSprop(block.parameters(), lr=3e-7)

            for x_m in X_hat_l:
                optimizer.zero_grad()
                with torch.no_grad():
                    y_teacher = block_orig(x_m)
                y_student = block(x_m)
                loss = torch.nn.functional.mse_loss(y_student, y_teacher)
                loss.backward()
                optimizer.step() 

        # Final pruning after RO
        final_score = compute_rgs_score(block, X_l, alpha)
        for name, param in block.named_parameters():
            if name in final_score:
                score = final_score[name]
                threshold = torch.quantile(score, 0.3)
                mask = (score > threshold).float()
                param.data *= mask

    return model
