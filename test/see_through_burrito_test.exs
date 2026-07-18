defmodule SeeThroughBurritoTest do
  use ExUnit.Case, async: true
  doctest SeeThroughBurrito

  test "version is set" do
    assert SeeThroughBurrito.version() == "0.1.0"
  end
end
