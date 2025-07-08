"""
Test the message utility functions.
"""

import mlx.core as mx
import numpy as np
import pytest

from parallax.p2p.message_util import (
    proto_to_request,
    proto_to_tensor,
    request_to_proto,
    tensor_to_proto,
)
from parallax.server.request import IntermediateRequest, RequestStatus


class TestMessageUtil:
    """
    Test the message utility functions.
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.request_id = "test_request_123"
        self.current_position = 10

    def test_request_to_proto_with_hidden_states(self):
        """Test converting IntermediateRequest with hidden_states to protobuf."""
        # Create hidden_states as mlx array (2D tensor)
        hidden_states = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)

        # Create IntermediateRequest
        request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
        )

        # Convert to protobuf - wrap in list
        forward_request = request_to_proto([request])

        # Verify the conversion
        assert len(forward_request.reqs) == 1
        proto_req = forward_request.reqs[0]
        assert proto_req.rid == self.request_id
        assert proto_req.output_length == self.current_position

        # Verify hidden_states are in pp_proxy_tensors
        assert len(forward_request.pp_proxy_tensors.tensors) == 1
        named_tensor = forward_request.pp_proxy_tensors.tensors[0]
        assert named_tensor.name == "hidden_states"

        # Verify tensor data - safetensor contains shape and dtype info
        tensor = named_tensor.tensor
        # Safetensor format doesn't set size and dtype in protobuf
        assert len(tensor.size) == 0  # size not set in safetensor format
        assert tensor.dtype == ""  # dtype not set in safetensor format

    def test_request_to_proto_with_token_id(self):
        """Test converting IntermediateRequest with single token_id to protobuf."""
        # Create hidden_states as single token_id (1D tensor with single element)
        token_id = 42
        hidden_states = mx.array([token_id], dtype=mx.int32)

        # Create IntermediateRequest
        request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.DECODING,
            hidden_states=hidden_states,
        )

        # Convert to protobuf - wrap in list
        forward_request = request_to_proto([request])

        # Verify the conversion
        assert len(forward_request.reqs) == 1
        proto_req = forward_request.reqs[0]
        assert proto_req.rid == self.request_id
        assert proto_req.output_length == self.current_position

        # Verify token_id is in next_token_ids, not in pp_proxy_tensors
        assert len(forward_request.next_token_ids) == 1
        assert forward_request.next_token_ids[0] == token_id
        assert len(forward_request.pp_proxy_tensors.tensors) == 0

    def test_request_to_proto_with_bfloat16(self):
        """Test converting IntermediateRequest with bfloat16 hidden_states."""
        # Create hidden_states as mlx array with bfloat16 dtype
        hidden_states = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)

        # Create IntermediateRequest
        request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
        )

        # Convert to protobuf - wrap in list
        forward_request = request_to_proto([request])

        # Verify the conversion
        assert len(forward_request.reqs) == 1
        proto_req = forward_request.reqs[0]
        assert proto_req.rid == self.request_id
        assert proto_req.output_length == self.current_position

        # Verify hidden_states are in pp_proxy_tensors
        assert len(forward_request.pp_proxy_tensors.tensors) == 1
        named_tensor = forward_request.pp_proxy_tensors.tensors[0]
        assert named_tensor.name == "hidden_states"

        # Verify tensor data - safetensor contains shape and dtype info
        tensor = named_tensor.tensor
        # Safetensor format doesn't set size and dtype in protobuf
        assert len(tensor.size) == 0  # size not set in safetensor format
        assert tensor.dtype == ""  # dtype not set in safetensor format

    def test_request_to_proto_assertion_error(self):
        """Test that assertion error is raised for non-mlx array hidden_states."""
        # Create IntermediateRequest with numpy array instead of mlx array
        hidden_states = np.array([[1.0, 2.0], [3.0, 4.0]])

        request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
        )

        # Should raise assertion error
        with pytest.raises(AssertionError):
            request_to_proto([request])

    def test_request_to_proto_multiple_requests_concatenation(self):
        """Test that multiple requests' hidden_states are concatenated into a single tensor."""
        # Create first request
        hidden_states1 = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        request1 = IntermediateRequest(
            request_id="req1",
            current_position=5,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states1,
        )
        
        # Create second request
        hidden_states2 = mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
        request2 = IntermediateRequest(
            request_id="req2",
            current_position=3,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states2,
        )
        
        # Create third request
        hidden_states3 = mx.array([[9.0, 10.0]], dtype=mx.float32)
        request3 = IntermediateRequest(
            request_id="req3",
            current_position=2,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states3,
        )

        # Convert to protobuf
        forward_request = request_to_proto([request1, request2, request3])

        # Verify that there are 3 requests
        assert len(forward_request.reqs) == 3
        
        # Verify that there's only one concatenated hidden_states tensor
        assert len(forward_request.pp_proxy_tensors.tensors) == 1
        named_tensor = forward_request.pp_proxy_tensors.tensors[0]
        assert named_tensor.name == "hidden_states"
        
        # Convert back to MLX tensor and verify concatenation
        concatenated_tensor = proto_to_tensor(named_tensor.tensor)
        
        # Expected concatenated tensor should be: [[1,2], [3,4], [5,6], [7,8], [9,10]]
        expected_concatenated = mx.concatenate([hidden_states1, hidden_states2, hidden_states3], axis=0)
        
        # Verify shape and values
        assert concatenated_tensor.shape == expected_concatenated.shape
        np.testing.assert_array_almost_equal(
            np.array(concatenated_tensor.tolist()),
            np.array(expected_concatenated.tolist())
        )
        
        # Verify that proto_to_request can correctly split the concatenated tensor back
        converted_requests = proto_to_request(forward_request)
        assert len(converted_requests) == 3
        
        # Verify that each request gets the correct portion of hidden_states
        # Note: The slicing logic depends on current_position, so we need to account for that
        for i, (original_req, converted_req) in enumerate(zip([request1, request2, request3], converted_requests)):
            assert converted_req.request_id == original_req.request_id
            assert converted_req.status == RequestStatus.PREFILLING
            # The hidden_states should be sliced based on current_position
            assert converted_req.hidden_states is not None

    def test_proto_to_request_with_hidden_states(self):
        """Test converting ForwardRequest protobuf back to IntermediateRequest with hidden_states."""
        # Create original request with multidimensional hidden_states
        hidden_states = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)
        original_request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.PREFILLING,
            hidden_states=hidden_states,
        )

        # Convert to protobuf and back - wrap in list and get first element
        proto_request = request_to_proto([original_request])
        converted_requests = proto_to_request(proto_request)
        converted_request = converted_requests[0]

        # Verify conversion
        assert converted_request.request_id == original_request.request_id
        # Note: current_position might be different due to len(proto_req.input_ids) calculation
        assert converted_request.hidden_states is not None

        # Verify hidden_states are the same (within numerical precision)
        original_np = np.array(original_request.hidden_states.tolist())
        converted_np = np.array(converted_request.hidden_states.tolist())
        np.testing.assert_array_almost_equal(original_np, converted_np)

    def test_proto_to_request_with_token_id(self):
        """Test converting ForwardRequest protobuf back to IntermediateRequest with token_id."""
        # Create original request with token_id
        token_id = 42
        hidden_states = mx.array([token_id], dtype=mx.int32)
        original_request = IntermediateRequest(
            request_id=self.request_id,
            current_position=self.current_position,
            status=RequestStatus.DECODING,
            hidden_states=hidden_states,
        )

        # Convert to protobuf and back - wrap in list and get first element
        proto_request = request_to_proto([original_request])
        converted_requests = proto_to_request(proto_request)
        converted_request = converted_requests[0]

        # Verify conversion
        assert converted_request.request_id == original_request.request_id
        # When hidden_states is a single token_id, it becomes None after conversion
        assert converted_request.hidden_states is None
        # The status changes to FINISHED_EOS
        assert converted_request.status == RequestStatus.FINISHED_EOS
        # The token_id is preserved in next_token_id
        assert converted_request.next_token_id == token_id

    @pytest.mark.parametrize(
        "original_tensor",
        [
            mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
            mx.array([1, 2, 3, 4], dtype=mx.int32),
            mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16),
            mx.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=mx.float32),
        ],
    )
    def test_tensor_to_proto_and_back(self, original_tensor):
        """Test direct tensor serialization and deserialization using safetensor."""
        # Convert to protobuf
        proto_tensor = tensor_to_proto(original_tensor)
        print(proto_tensor)
        # Convert back to MLX array
        converted_tensor = proto_to_tensor(proto_tensor)

        # Verify shape and dtype are preserved
        assert converted_tensor.shape == original_tensor.shape
        assert converted_tensor.dtype == original_tensor.dtype

        # Verify values are the same (within numerical precision)
        original_np = np.array(original_tensor.tolist())
        converted_np = np.array(converted_tensor.tolist())
        np.testing.assert_array_almost_equal(original_np, converted_np)

    def test_safetensor_preserves_dtype(self):
        """Test that safetensor serialization preserves exact dtype."""
        # Test with bfloat16 which was problematic with the old numpy approach
        original_tensor = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)

        # Convert to protobuf and back
        proto_tensor = tensor_to_proto(original_tensor)
        converted_tensor = proto_to_tensor(proto_tensor)

        # Verify dtype is exactly preserved
        assert converted_tensor.dtype == mx.bfloat16
        assert converted_tensor.dtype == original_tensor.dtype

        # Verify the tensor is actually bfloat16
        assert converted_tensor.dtype == mx.bfloat16

    @pytest.mark.parametrize(
        "dtype,shape",
        [
            (mx.float32, ()),
            (mx.float32, (1,)),
            (mx.float32, (2, 2)),
            (mx.float32, (3, 1, 4)),
            (mx.float32, (10, 10)),
            (mx.bfloat16, (2, 2)),
            (mx.bfloat16, (1,)),
            (mx.int32, (5,)),
            (mx.int32, (2, 3)),
            (mx.int64, (1, 1)),
            (mx.int64, (4, 2)),
        ],
    )
    def test_tensor_to_proto_and_back_various_dtypes_and_shapes(self, dtype, shape):
        """Test MLX array safetensor序列化/反序列化的类型和shape一致性。"""
        if shape == ():
            arr = mx.array(42, dtype=dtype)
        else:
            arr = mx.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

        proto = tensor_to_proto(arr)
        assert proto.buffer
        arr2 = proto_to_tensor(proto)
        assert isinstance(arr2, mx.array)
        assert arr2.shape == arr.shape
        assert arr2.dtype == arr.dtype
        np.testing.assert_array_equal(np.array(arr.tolist()), np.array(arr2.tolist()))

    @pytest.mark.parametrize(
        "arr,expected_shape,expected_dtype,expected_value",
        [
            (mx.array([123], dtype=mx.int32), (1,), mx.int32, 123),
            (mx.array([42.5], dtype=mx.float32), (1,), mx.float32, 42.5),
        ],
    )
    def test_tensor_to_proto_and_back_empty_and_singleton(
        self, arr, expected_shape, expected_dtype, expected_value
    ):
        """Test空tensor和单元素tensor的序列化/反序列化。"""
        proto = tensor_to_proto(arr)
        arr2 = proto_to_tensor(proto)
        assert arr2.shape == expected_shape
        assert arr2.dtype == expected_dtype
        if expected_value is not None:
            assert float(arr2[0]) == expected_value

    @pytest.mark.parametrize("size", [1024, 1024 * 1024, 1024 * 1024 * 10])
    def test_tensor_to_proto_and_back_large_tensor(self, size):
        """Test大tensor序列化/反序列化。"""
        arr = mx.arange(size, dtype=mx.float32)
        proto = tensor_to_proto(arr)
        arr2 = proto_to_tensor(proto)
        assert arr2.shape == arr.shape
        assert arr2.dtype == arr.dtype
        np.testing.assert_array_equal(np.array(arr.tolist()), np.array(arr2.tolist()))

    @pytest.mark.parametrize(
        "hidden_states,expected_status",
        [
            (mx.array([[1.1, 2.2], [3.3, 4.4]], dtype=mx.float32), RequestStatus.PREFILLING),
            (
                mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=mx.bfloat16),
                RequestStatus.PREFILLING,
            ),
            (mx.array([1, 2, 3, 4], dtype=mx.int32), RequestStatus.PREFILLING),
            (mx.array([42], dtype=mx.int32), RequestStatus.FINISHED_EOS),
        ],
    )
    def test_request_to_proto_and_back_end_to_end(self, hidden_states, expected_status):
        """Test IntermediateRequest端到端序列化/反序列化一致性。"""
        status = RequestStatus.PREFILLING if hidden_states.shape != (1,) else RequestStatus.DECODING
        req = IntermediateRequest(
            request_id="req42",
            current_position=7,
            status=status,
            hidden_states=hidden_states,
        )
        proto = request_to_proto([req])
        requests = proto_to_request(proto)
        req2 = requests[0]
        assert req2.request_id == req.request_id
        assert req2.status == expected_status
        if expected_status == RequestStatus.FINISHED_EOS:
            # For single token, hidden_states becomes None and token is in next_token_id
            assert req2.hidden_states is None
            assert req2.next_token_id == int(hidden_states[0])
        else:
            # For multi-dimensional hidden_states, they should be preserved
            assert req2.hidden_states is not None
            np.testing.assert_array_almost_equal(
                np.array(req.hidden_states.tolist()), np.array(req2.hidden_states.tolist())
            )

    @pytest.mark.parametrize("token_id", [0, 42, 99, 1000, 99999])
    def test_request_to_proto_and_back_token_id(self, token_id):
        """Test单token id的request序列化/反序列化。"""
        req = IntermediateRequest(
            request_id="tokreq",
            current_position=1,
            status=RequestStatus.DECODING,
            hidden_states=mx.array([token_id], dtype=mx.int32),
        )
        proto = request_to_proto([req])
        requests = proto_to_request(proto)
        req2 = requests[0]
        # When hidden_states is a single token_id, it becomes None after conversion
        assert req2.hidden_states is None
        # The token_id is preserved in next_token_id
        assert req2.next_token_id == token_id
        # The status changes to FINISHED_EOS
        assert req2.status == RequestStatus.FINISHED_EOS
