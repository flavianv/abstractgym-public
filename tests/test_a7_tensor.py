"""Optional Torch tests: loss direction, token alignment, and stop-gradient."""
import pytest
torch=pytest.importorskip('torch')
from abstractgym.a7 import forward_kl

def test_forward_kl_and_teacher_stop_gradient():
    student=torch.tensor([[1.,-1.,0.],[0.,2.,1.]],requires_grad=True)
    teacher=torch.tensor([[0.,2.,1.],[2.,0.,1.]],requires_grad=True)
    q=teacher.detach().softmax(-1)
    expected=(q*(q.log()-student.log_softmax(-1))).sum(-1).mean()
    loss=forward_kl(student,teacher)
    assert torch.allclose(loss,expected)
    loss.backward()
    assert teacher.grad is None
    assert torch.allclose(student.grad,(student.detach().softmax(-1)-q)/2,atol=1e-7)
    assert forward_kl(teacher,teacher).abs()<1e-6

def test_response_logit_slice_predicts_same_generated_prefix():
    # At token t a causal LM predicts t+1; teacher and student prompts differ in length.
    response=[7,8,9]
    for prompt in ([1,2],[1,2,3,4,5]):
        sequence=prompt+response
        positions=torch.arange(len(sequence))[-(len(response)+1):][:-1]
        assert [sequence[i+1] for i in positions]==response
