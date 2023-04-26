#include "overload_smooth.hpp"

// branchTape bt = branchTape();
template <typename T> branchTape<T>::branchTape() {
  contribution = 1;
  markers = std::vector<marker>();
  markerIndexCounter = 0;
  last_falsified_marker_pos = -1;
}

template <typename T> bool branchTape<T>::next_round() {
  /* returns: true if another round necessary, false if it's the last round! */
  // consume all consecutive falsified ones from the bottom up
  while (!markers.empty() && markers.back().notFalsified == false) {
    // markers left,
    markers.pop_back();
  }
  // no more left: we're done with taking contributions
  if (markers.empty()) {
    return false;
  } else { // some left: take the other path! (we have bottom markers left which
           // aren't falsified)
    // next round goes with falsifying last one!
    markers.back().notFalsified = false;
    last_falsified_marker_pos = markers.back().counterPosition;
    // std::cout << "reset to" << last_falsified_marker_pos << std::endl;
  }
  // reset all running variables to code-beginning: contribution,
  contribution = 1;
  markerIndexCounter = 0;
  return true;
}

template <typename T>
bool branchTape<T>::check(const bool &absolute, const T &tendency) {
  /* Called by each overloaded operator.
   *   EVAL PHASE: going down the markers to the last marked condition,
   *    whose marker was newly falsified at the end of the last round.
   *   - if marker not falsified, return bool record contribution as usual
   *   - if marker falsified, return inverse bool and multiply with (1 -
   * contribution) RECORD PHASE: record new markers. For each marked condition:
   *   - give back bool as evaluated normally, record contribution
   *   - record marker with its count and boolean value
   */

  bool new_truthValue; // either original, or inversed if falsified

  double intpart;
  if (modf(dco::passive_value(tendency), &intpart) ==
      0.0) { // tendency numerically 1 or 0 => no marker set
    return absolute;
  } else { // modf(tendency, &intpart) != 0.0
    if (markerIndexCounter > last_falsified_marker_pos) {
      // PAST LAST MARKER: RECORDING PHASE
      // tendency numerically inverval (1,0) && we're beyond last falsified
      // we're below the marker counter, so we just go along the branches

      markers.push_back(marker{markerIndexCounter, true});
      contribution = contribution * tendency;
      new_truthValue = absolute;
    } else {
      // markercounter passed? we have to record a new one
      marker &m = markers[markerIndexCounter];

      new_truthValue = absolute ^ !m.notFalsified;

      // ... and invert the contribution
      contribution = contribution * pow(tendency, (int)m.notFalsified) *
                     pow(1 - tendency, (int)!m.notFalsified);
    }
    markerIndexCounter++;
    return new_truthValue;
  }
}

template <typename T>
branchTape<T> global_branchtape<T>::bt; // define the member

template <typename T> bool global_branchtape<T>::next_round() {
  return global_branchtape::bt.next_round();
}

template <typename T>
bool global_branchtape<T>::check(const bool &absolute, const T &tendency) {
  return global_branchtape::bt.check(absolute, tendency);
}

template <typename T> void global_branchtape<T>::new_tape() {
  global_branchtape::bt = branchTape<T>();
}

template <typename T> T &global_branchtape<T>::contribution() {
  return global_branchtape::bt.contribution;
}

#define EXPLICITLY_INSTANTIATE(T)                                              \
  template class global_branchtape<T>;                                         \
  // already initializes all its methods!

EXPLICITLY_INSTANTIATE(DCO_TanType<double>);
EXPLICITLY_INSTANTIATE(DCO_T<double>);
EXPLICITLY_INSTANTIATE(double);

EXPLICITLY_INSTANTIATE(DCO_TanType<float>);
EXPLICITLY_INSTANTIATE(DCO_T<float>);
EXPLICITLY_INSTANTIATE(float);
