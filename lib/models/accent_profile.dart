class AccentProfile {
  final List<double> vector;

  AccentProfile(this.vector);

  Map<String, dynamic> toJson() => {
    'vector': vector,
  };

  factory AccentProfile.fromJson(Map<String, dynamic> json) {
    return AccentProfile(List<double>.from(json['vector']));
  }
}
