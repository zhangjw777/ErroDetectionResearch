"""
测试新增模块的功能
"""
import torch
import sys

def main():
    print('=' * 60)
    print('Testing Module Imports...')
    print('=' * 60)
    
    # Test imports
    from modeling import SyntaxSemanticInteractionLayer, ErrorAwareSentenceHead, GECModelWithMTL
    from loss import UncertaintyWeightedLoss, MultiTaskLossWithUncertainty
    
    print('[OK] All imports successful')
    
    # Test SyntaxSemanticInteractionLayer
    print('\n' + '=' * 60)
    print('Testing SyntaxSemanticInteractionLayer...')
    print('=' * 60)
    
    batch_size, seq_len, hidden_size = 2, 32, 768
    h_shared = torch.randn(batch_size, seq_len, hidden_size)
    h_svo = torch.randn(batch_size, seq_len, hidden_size)
    
    fusion_layer = SyntaxSemanticInteractionLayer(hidden_size)
    h_gec_input = fusion_layer(h_shared, h_svo)
    
    print(f'Input shape: h_shared={h_shared.shape}, h_svo={h_svo.shape}')
    print(f'Output shape: h_gec_input={h_gec_input.shape}')
    assert h_gec_input.shape == h_shared.shape, 'Shape mismatch!'
    print('[OK] SyntaxSemanticInteractionLayer test passed')
    
    # Test ErrorAwareSentenceHead
    print('\n' + '=' * 60)
    print('Testing ErrorAwareSentenceHead...')
    print('=' * 60)
    
    num_gec_labels = 100
    gec_logits = torch.randn(batch_size, seq_len, num_gec_labels)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -5:] = 0  # Simulate padding
    
    sent_head = ErrorAwareSentenceHead(hidden_size)
    sent_logits, attn_weights = sent_head(h_gec_input, gec_logits, attention_mask)
    
    print(f'Input: h_tokens={h_gec_input.shape}, gec_logits={gec_logits.shape}')
    print(f'Output: sent_logits={sent_logits.shape}, attn_weights={attn_weights.shape}')
    assert sent_logits.shape == (batch_size, 2), 'Sent logits shape mismatch!'
    assert attn_weights.shape == (batch_size, seq_len), 'Attention weights shape mismatch!'
    print('[OK] ErrorAwareSentenceHead test passed')
    
    # Test UncertaintyWeightedLoss
    print('\n' + '=' * 60)
    print('Testing UncertaintyWeightedLoss...')
    print('=' * 60)
    
    loss_gec = torch.tensor(2.5)
    loss_svo = torch.tensor(1.2)
    loss_sent = torch.tensor(0.8)
    
    uwl = UncertaintyWeightedLoss()
    total_loss, loss_dict = uwl(loss_gec, loss_svo, loss_sent)
    
    print(f'Input losses: GEC={loss_gec.item():.2f}, SVO={loss_svo.item():.2f}, SENT={loss_sent.item():.2f}')
    print(f'Total weighted loss: {total_loss.item():.4f}')
    print(f"Task weights: GEC={loss_dict['weight_gec']:.3f}, SVO={loss_dict['weight_svo']:.3f}, SENT={loss_dict['weight_sent']:.3f}")
    assert total_loss.dim() == 0, 'Total loss should be scalar!'
    print('[OK] UncertaintyWeightedLoss test passed')
    
    # Test MultiTaskLossWithUncertainty
    print('\n' + '=' * 60)
    print('Testing MultiTaskLossWithUncertainty...')
    print('=' * 60)
    
    num_svo_labels = 7
    gec_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    gec_labels[0, 10] = 5  # 模拟错误
    svo_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    sent_labels = torch.tensor([1, 0], dtype=torch.long)
    svo_logits = torch.randn(batch_size, seq_len, num_svo_labels)
    
    mtl_loss = MultiTaskLossWithUncertainty(
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_uncertainty_weighting=True
    )
    
    total_loss, loss_dict = mtl_loss(
        gec_logits, svo_logits, sent_logits,
        gec_labels, svo_labels, sent_labels,
        label_mask=None
    )
    
    print(f'Total loss: {total_loss.item():.4f}')
    print(f"GEC loss: {loss_dict['loss_gec']:.4f}")
    print(f"SVO loss: {loss_dict['loss_svo']:.4f}")
    print(f"Sent loss: {loss_dict['loss_sent']:.4f}")
    print('[OK] MultiTaskLossWithUncertainty test passed')
    
    print('\n' + '=' * 60)
    print('ALL TESTS PASSED!')
    print('=' * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
