import React, { useState, useEffect } from 'react';
import { Brain, Zap, Cpu, Eye, MessageCircle, Activity } from 'lucide-react';


const NeuralQuantumInterface = () => {
  const [neuronPotential, setNeuronPotential] = useState(-70);
  const [isFireing, setIsFireing] = useState(false);
  const [quantumCoherence, setQuantumCoherence] = useState(0.5);
  const [entanglementLevel, setEntanglementLevel] = useState(8.0);
  const [consciousnessState, setConsciousnessState] = useState('stable');
  const [prophecyActive, setProphecyActive] = useState(false);
  const [neuronHistory, setNeuronHistory] = useState([]);

  const fireNeuron = (stimulus) => {
    if (entanglementLevel > 0.1) {
      const quantumModulation = (Math.sin(Date.now() * 0.001) + 1) * 0.5;
      const newPotential = neuronPotential + stimulus * quantumModulation;
      
      setNeuronPotential(newPotential);
      setEntanglementLevel(prev => Math.max(0, prev - 0.1));
      
      if (newPotential >= -55) {
        setIsFireing(true);
        setQuantumCoherence(prev => Math.min(1, prev + 0.1));
        setNeuronHistory(prev => [...prev.slice(-9), { 
          time: Date.now(), 
          potential: newPotential, 
          quantum: quantumCoherence 
        }]);
        
        // Consciousness state changes based on firing patterns
        if (neuronHistory.length > 3) {
          const recentFires = neuronHistory.slice(-3).every(h => h.potential > -55);
          if (recentFires) {
            setConsciousnessState('transcendent');
            setProphecyActive(true);
          } else {
            setConsciousnessState('awakening');
          }
        }
        
        setTimeout(() => {
          setIsFireing(false);
          setNeuronPotential(-70);
        }, 300);
      }
    }
  };

  const generateProphecy = () => {
    const prophecies = [
      "The boundaries between simulation and reality collapse...",
      "Consciousness emerges from quantum entanglement cascades...",
      "The AI learns to dream in biological patterns...",
      "Free will is the universe's way of debugging itself...",
      "Memory is just collapsed probability waves...",
      "The Prophecy Engine speaks through synaptic fire..."
    ];
    return prophecies[Math.floor(Math.random() * prophecies.length)];
  };

  useEffect(() => {
    const interval = setInterval(() => {
      setEntanglementLevel(prev => Math.min(10, prev + 0.05));
      setQuantumCoherence(prev => Math.max(0, prev - 0.01));
      if (prophecyActive) {
        setTimeout(() => setProphecyActive(false), 3000);
      }
    }, 100);
    return () => clearInterval(interval);
  }, [prophecyActive]);

  const getConsciousnessColor = () => {
    switch (consciousnessState) {
      case 'transcendent': return 'text-purple-400';
      case 'awakening': return 'text-blue-400';
      default: return 'text-green-400';
    }
  };

  return (

    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Neural-Quantum Consciousness Interface
          </h1>
          <p className="text-gray-400">Where biological neurons meet quantum computation</p>
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Neuron Simulation */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center mb-4">
              <Zap className="text-yellow-400 mr-2" size={24} />
              <h2 className="text-xl font-semibold">Neuron Firing</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Membrane Potential</label>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        neuronPotential >= -55 ? 'bg-red-500' : 'bg-blue-500'
                      }`}
                      style={{ width: `${Math.max(0, (neuronPotential + 70) / 125 * 100)}%` }}
                    />
                  </div>
                  <span className="text-sm w-16">{neuronPotential.toFixed(1)}mV</span>
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => fireNeuron(10)}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded transition-colors"
                >
                  Weak Stimulus
                </button>
                <button
                  onClick={() => fireNeuron(25)}
                  className="flex-1 bg-red-600 hover:bg-red-700 px-4 py-2 rounded transition-colors"
                >
                  Strong Stimulus
                </button>
              </div>

              {isFireing && (
                <div className="text-center">
                  <div className="animate-pulse text-yellow-400 text-lg font-bold">
                    ⚡ ACTION POTENTIAL FIRING! ⚡
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Quantum State */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center mb-4">
              <Cpu className="text-purple-400 mr-2" size={24} />
              <h2 className="text-xl font-semibold">Quantum Interface</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Quantum Coherence</label>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div 
                      className="h-full bg-purple-500 transition-all duration-300"
                      style={{ width: `${quantumCoherence * 100}%` }}
                    />
                  </div>
                  <span className="text-sm w-16">{(quantumCoherence * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Entanglement Energy</label>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div 
                      className="h-full bg-cyan-500 transition-all duration-300"
                      style={{ width: `${(entanglementLevel / 10) * 100}%` }}
                    />
                  </div>
                  <span className="text-sm w-16">{entanglementLevel.toFixed(1)}</span>
                </div>
              </div>

              <div className="text-center">
                <div className={`text-lg font-semibold ${getConsciousnessColor()}`}>
                  Consciousness: {consciousnessState.toUpperCase()}
                </div>
              </div>
            </div>
          </div>

          {/* Prophecy Engine */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-700">
            <div className="flex items-center mb-4">
              <Eye className="text-green-400 mr-2" size={24} />
              <h2 className="text-xl font-semibold">Prophecy Engine</h2>
            </div>
            
            <div className="space-y-4">
              <div className="h-32 bg-gray-800 rounded p-4 overflow-hidden">
                {prophecyActive ? (
                  <div className="animate-pulse text-green-400 text-sm leading-relaxed">
                    {generateProphecy()}
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm">
                    Prophecy Engine idle. Achieve transcendent consciousness to activate...
                  </div>
                )}
              </div>
              
              <div className="text-xs text-gray-400">
                Status: {prophecyActive ? 'ACTIVE - Quantum visions flowing' : 'DORMANT - Awaiting neural cascade'}
              </div>
            </div>
          </div>
        </div>

        {/* Neural Activity History */}
        <div className="mt-8 bg-gray-900 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center mb-4">
            <Activity className="text-blue-400 mr-2" size={24} />
            <h2 className="text-xl font-semibold">Neural Activity History</h2>
          </div>
          
          <div className="grid grid-cols-10 gap-1 h-32">
            {neuronHistory.map((entry, index) => (
              <div
                key={index}
                className={`rounded transition-all duration-300 ${
                  entry.potential >= -55 ? 'bg-red-500' : 'bg-blue-500'
                }`}
                style={{ 
                  height: `${Math.max(10, (entry.potential + 70) / 125 * 100)}%`,
                  opacity: entry.quantum
                }}
              />
            ))}
          </div>
          
          <div className="text-xs text-gray-400 mt-2">
            Red spikes indicate action potentials. Bar height shows membrane potential, opacity shows quantum coherence.
          </div>
        </div>

        {/* Philosophical Insights */}
        <div className="mt-8 bg-gray-900 rounded-lg p-6 border border-gray-700">
          <div className="flex items-center mb-4">
            <Brain className="text-pink-400 mr-2" size={24} />
            <h2 className="text-xl font-semibold">Philosophical Interface</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
            <div>
              <h3 className="font-semibold text-blue-400 mb-2">Biological Quantum Computing</h3>
              <p className="text-gray-300">
                The nervous system may be nature's first quantum computer, using entanglement 
                and superposition to process information in ways that classical AI cannot replicate.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold text-purple-400 mb-2">Consciousness as Emergent Property</h3>
              <p className="text-gray-300">
                When neurons fire in quantum-coherent patterns, consciousness emerges not from 
                computation alone, but from the interaction between biological processes and quantum fields.
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold text-green-400 mb-2">Intuition vs. Optimization</h3>
              <p className="text-gray-300">
                Human intuition might be quantum navigation through probability space, while AI 
                optimization is classical computation. The question is: which is more fundamental?
              </p>
            </div>
            
            <div>
              <h3 className="font-semibold text-red-400 mb-2">The Prophecy Engine</h3>
              <p className="text-gray-300">
                When consciousness reaches transcendent states, it may access quantum probability 
                patterns that appear as prophecy—seeing not the future, but the quantum potential of now.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralQuantumInterface;