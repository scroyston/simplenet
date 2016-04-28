package com.rgm.simpletest

class NegSquare {
  var lastx = 0.0
  def forward(x: Double): Double = { lastx = x; -math.pow(x, 2) }

  def backward(derr: Double): Double = { -2*lastx*derr }
}

class SimpleLinear(startWeights: Array[Double]) {
  val weights = {
    val tmp = new Array[Double](startWeights.length + 1)
    System.arraycopy(startWeights, 0, tmp, 0, startWeights.length)
    tmp
  }

  val lastInputs = new Array[Double](weights.length)
  lastInputs(lastInputs.length-1) = 1.0

  val grads = new Array[Double](weights.length)
  val inputGrads = new Array[Double](startWeights.length)

  def forward(inputs: Array[Double]): Double = {
    System.arraycopy(inputs, 0, lastInputs, 0, inputs.length)
    var sum = 0.0
    var i = 0
    while (i < weights.length) {
      sum += weights(i) * lastInputs(i)
      i += 1
    }
    sum
  }

  def backward(derr: Double): Array[Double] = {
    var i = 0
    while (i < weights.length) {
      grads(i) = derr * lastInputs(i)
      i += 1
    }
    i = 0
    while (i < inputGrads.length) {
      inputGrads(i) = derr * weights(i)
      i += 1
    }
    inputGrads
  }

  def update(stepSize: Double): Unit = {
    var i = 0
    while (i < weights.length) {
      weights(i) += stepSize * grads(i)
      i += 1
    }
  }
}

class SimpleCritic(weights: Array[Double]) {
  val p1 = new SimpleLinear(weights)
  val p2 = new NegSquare()

  def forward(inputs: Array[Double]): Double = {
    val x = p1.forward(inputs)
    p2.forward(x)
  }

  def backward(derr: Double): Array[Double] = {
    val err2 = p2.backward(derr)
    p1.backward(err2)
  }

  def update(stepSize: Double): Unit = { p1.update(stepSize) }
}

class FullThing(cweights: Array[Double], aweights: Array[Double]) {
  val gamma = 0.95
  val tau = 0.001
  val stepSize = 0.001
  val critic = new SimpleCritic(cweights)
  val target_critic = new SimpleCritic(cweights)
  val actor = new SimpleLinear(aweights)
  val target_actor = new SimpleLinear(aweights)
  val ainputs = new Array[Double](1)
  val cinputs = new Array[Double](2)
  val rand = new java.util.Random()

  def act(s0: Double): Double = {
    val action = actor.forward(Array(s0))
    if (rand.nextDouble() > 0.8) {
      action + rand.nextGaussian()
    }
    else action
  }

  def train(s0: Double, a0: Double, r0: Double, s1: Double): Unit = {
    val qval0 = critic.forward(Array(s0, a0))
    val a1 = target_actor.forward(Array(s1))

    val qval1 = target_critic.forward(Array(s1, a1))
    val qest = r0 + gamma * qval1

    val qdiff = qval0 - qest
    val derr_qval = -(2 * qest - 2*qval0)

    critic.backward(derr_qval)
    critic.update(stepSize)

    val a0new = actor.forward(Array(s0))
    critic.forward(Array(s0, a0new))
    val inGrad = critic.backward(1)
    actor.backward(inGrad(1))
    actor.update(stepSize)
    update_target(target_actor.weights, actor.weights)
    update_target(target_critic.p1.weights, critic.p1.weights)
  }

  def update_target(tweights: Array[Double], weights: Array[Double]): Unit = {
    var i = 0
    while (i < tweights.length) {
      tweights(i) = (1.0 - tau) * tweights(i) + tau * weights(i)
      i += 1
    }
  }
}

object Runner {
  def main(args: Array[String]) {
    go()
  }

  def go(): Unit = {
    val rand = new java.util.Random()
    val cweights = Array(rand.nextDouble() * 0.006 - 0.003, rand.nextDouble() * 0.006 - 0.003)
    val aweights = Array(rand.nextDouble() * 0.006 - 0.003)
    val tmp = new FullThing(cweights, aweights)

    val history = new Array[Array[Double]](30000)
    var historyLen = 0
    var historyIdx = 0

    var i = 0
    while (i < 10000) {
      val state = rand.nextDouble()
      val action = tmp.act(state)
      val reward = -math.pow(3.7*state - action, 2)

      history(historyIdx) = Array(state, action, reward)
      historyIdx = (historyIdx + 1) % history.length
      if (historyLen < history.length) { historyLen += 1 }

      if (historyLen > 16) {
        val idx = rand.nextInt(historyLen)
        val entry = history(idx)
        tmp.train(entry(0), entry(1), entry(2), rand.nextDouble())
      }
      println(tmp.critic.p1.weights.mkString(" ") + " " + tmp.actor.weights.mkString(" "))
      i += 1
    }
  }
}

class SimpleNet {
    var w1 = 0.0
    var w2 = 0.0
    var w3 = 0.0
    var w4 = 0.0
    var w5 = 0.0
    var x1 = 0.0
    var y1 = 0.0
    var actual = 0.0

  def foo(): Unit = {


    val z1 = w1 * x1 + w2 * y1 + w3
    val z2 = z1 * z1
    val z3 = w4 * z2 + w5

    val E = (actual - z3) * (actual - z3)


    val d_E_wrt_z3 = -2 * actual + 2 * z3

    val d_z3_wrt_z2 = w4
    val d_z3_wrt_w4 = z2
    val d_z3_wrt_w5 = 1

    val d_z2_wrt_z1 = 2 * z1

    val d_z1_wrt_x1 = w1
    val d_z1_wrt_w1 = x1
    val d_z1_wrt_y1 = w2
    val d_z1_wrt_w2 = y1
    val d_z1_wrt_w3 = 1


    val d_E_wrt_z2 = d_E_wrt_z3 * d_z3_wrt_z2  // (-2 * actual + 2 * z3) * w4
    val d_E_wrt_w4 = d_E_wrt_z3 * d_z3_wrt_w4   // (-2 * actual + 2 * z3) * z2
    val d_E_wrt_w5 = d_E_wrt_z3 * d_z3_wrt_w5   // (-2 * actual + 2 * z3) * 1


    val d_E_wrt_z1 = d_E_wrt_z2 * d_z2_wrt_z1 //  (-2 * actual + 2 * z3) * w4 * (2 * z1)

    val d_E_wrt_x1 = d_E_wrt_z1 * d_z1_wrt_x1 //  (-2 * actual + 2 * z3) * w4 * (2 * z1) * (w1)
    // (-2 * actual + 2 * (w4 * ((w1*x1 + w2*y1 + w3) * (w1*x1 + w2*y1 + w3)) + w5) * w4 * (2 * (w1*x1 + w2*y1 + w3)) * w1



    val d_E_wrt_w1 = d_E_wrt_z1 * d_z1_wrt_w1 //  (-2 * actual + 2 * z3) * w4 * (2 * z1) * (x1)
    val d_E_wrt_y1 = d_E_wrt_z1 * d_z1_wrt_y1 //  (-2 * actual + 2 * z3) * w4 * (2 * z1) * (w2)
    val d_E_wrt_w2 = d_E_wrt_z1 * d_z1_wrt_w2 //  (-2 * actual + 2 * z3) * w4 * (2 * z1) * (y1)
    val d_E_wrt_w3 = d_E_wrt_z1 * d_z1_wrt_w3 //  (-2 * actual + 2 * z3) * w4 * (2 * z1) * 1








  }
}
